"""A rule-based responder that lets the fake provider hold a sensible conversation.

It plays every role in the graph without a language model:

* for the router it classifies the latest message by keyword (ties go to the earlier
  category in :data:`PRIORITY`) and answers through the ``RouteDecision`` tool call;
* for a specialist it reads the ``Role:`` line of the system prompt and the tool
  results of the current turn, then either calls the next tool or writes the reply.

It is deliberately simple. It exists so the API, the CLI and the tests can exercise the
real graph, tools, approvals and handoffs with zero credentials.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from support_agent.graph import TRANSFER_TOOL
from support_agent.llm import ai_tool_calls, bound_tool_names, tool_call
from support_agent.router import RouteDecision
from support_agent.state import Category
from support_agent.tools import (
    CHECK_REFUND_ELIGIBILITY,
    CREATE_SUPPORT_TICKET,
    ISSUE_REFUND,
    LIST_CUSTOMER_ORDERS,
    LOOKUP_ORDER,
    SEARCH_KNOWLEDGE_BASE,
)

ROUTE_TOOL = RouteDecision.__name__

_ORDER_ID = re.compile(r"\bORD-\d{5}\b", re.IGNORECASE)
_AMOUNT = re.compile(r"\$\s?(\d+(?:\.\d{1,2})?)")
_ROLE = re.compile(r"^Role: (\w+)", re.MULTILINE)
_CUSTOMER = re.compile(r"Customer of this conversation: (CUST-\d{4})")

# fmt: off
KEYWORDS: dict[Category, tuple[str, ...]] = {
    "escalate": (
        "human", "real person", "manager", "supervisor", "complaint", "lawyer", "legal",
        "unacceptable", "speak to", "talk to someone",
    ),
    "order_status": (
        "where is", "where's", "track", "tracking", "shipped", "shipping", "delivery",
        "delivered", "arrive", "arrived", "status", "cancel", "late",
    ),
    "billing": (
        "refund", "refunded", "money back", "charge", "charged", "charges", "payment",
        "invoice", "bill", "billing",
    ),
    "technical": (
        "password", "log in", "login", "sign in", "app", "error", "crash", "crashes", "sync",
        "two-factor", "2fa", "authenticator", "account", "bug", "website",
    ),
}
# fmt: on
PRIORITY: tuple[Category, ...] = ("escalate", "order_status", "billing", "technical")
_REFUND_WORDS = ("refund", "money back")


def _matches(text: str, phrases: Sequence[str]) -> list[str]:
    lowered = text.lower()
    return [p for p in phrases if re.search(rf"\b{re.escape(p)}\b", lowered)]


def classify(text: str) -> tuple[Category, list[str]]:
    """Pick a category by keyword count; returns the category and the matched keywords."""
    hits = {category: _matches(text, KEYWORDS[category]) for category in PRIORITY}
    best = max(PRIORITY, key=lambda c: (len(hits[c]), -PRIORITY.index(c)))
    if not hits[best]:
        return ("order_status" if _ORDER_ID.search(text) else "technical"), []
    return best, hits[best]


@dataclass
class _Turn:
    """What a specialist knows: the request, its role and this turn's tool results."""

    request: str
    role: str | None
    customer_id: str | None
    results: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    @property
    def order_id(self) -> str | None:
        match = _ORDER_ID.search(self.request)
        return match.group(0).upper() if match else None

    @property
    def amount(self) -> str | None:
        match = _AMOUNT.search(self.request)
        return match.group(1) if match else None

    @property
    def wants_refund(self) -> bool:
        return bool(_matches(self.request, _REFUND_WORDS))

    @property
    def transferred(self) -> bool:
        return bool(self.results.get(TRANSFER_TOOL))

    def last(self, tool: str) -> dict[str, Any] | None:
        results = self.results.get(tool)
        return results[-1] if results else None


def _read_turn(messages: Sequence[BaseMessage]) -> _Turn:
    system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
    role = _ROLE.search(system)
    customer = _CUSTOMER.search(system)
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    turn = _Turn(
        request=str(messages[start].content) if start >= 0 else "",
        role=role.group(1) if role else None,
        customer_id=customer.group(1) if customer else None,
    )
    for message in messages[start + 1 :]:
        if isinstance(message, ToolMessage) and message.name:
            try:
                payload = json.loads(str(message.content))
            except json.JSONDecodeError:
                payload = {"error": str(message.content)}
            turn.results[message.name].append(payload)
    return turn


def _call(name: str, **args: Any) -> AIMessage:
    return ai_tool_calls(tool_call(name, args))


def _describe_order(order: dict[str, Any], *, wants_cancel: bool) -> str:
    order_id, status = order["order_id"], order["status"]
    items = ", ".join(item["name"] for item in order["items"])
    if status == "processing":
        text = f"Order {order_id} ({items}) is being prepared and has not shipped yet."
        if wants_cancel:
            text += " It can still be cancelled: open Orders, choose the order and select Cancel."
        return text
    if status == "shipped":
        return (
            f"Order {order_id} ({items}) is on its way with {order['carrier']}. "
            f"The tracking number is {order['tracking_number']}."
        )
    if status == "delivered":
        return f"Order {order_id} ({items}) was delivered on {order['delivered_on']}."
    return f"Order {order_id} ({items}) was {status}."


def _subject(request: str) -> str:
    subject = " ".join(request.split())[:80]
    return subject if len(subject) >= 3 else "Customer request"


class DemoPolicy:
    """The :data:`~support_agent.llm.Responder` used by the ``fake`` provider by default."""

    def __call__(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
    ) -> AIMessage:
        if ROUTE_TOOL in bound_tool_names(tools):
            return self._route(messages)
        turn = _read_turn(messages)
        handlers = {
            "billing": self._billing,
            "order_status": self._order_status,
            "technical": self._technical,
            "escalation": self._escalation,
        }
        handler = handlers.get(turn.role or "")
        if handler is None:
            return AIMessage(content="I can help with orders, refunds and account questions.")
        return handler(turn)

    @staticmethod
    def _route(messages: Sequence[BaseMessage]) -> AIMessage:
        request = next(
            (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        category, matched = classify(request)
        reason = f"Matched keywords: {', '.join(matched)}." if matched else "No keyword matched."
        return _call(ROUTE_TOOL, category=category, reason=reason)

    # -- specialists ----------------------------------------------------------------------

    def _billing(self, turn: _Turn) -> AIMessage:
        if not turn.wants_refund:
            return self._answer_from_kb(turn, ticket_category=None)
        order_id = turn.order_id
        if order_id is None:
            return self._ask_for_order(turn, purpose="refunded")

        eligibility = turn.last(CHECK_REFUND_ELIGIBILITY)
        if eligibility is None:
            args: dict[str, Any] = {"order_id": order_id}
            if turn.amount is not None:
                args["amount"] = turn.amount
            return _call(CHECK_REFUND_ELIGIBILITY, **args)
        if "error" in eligibility:
            return AIMessage(content=f"I couldn't check that order: {eligibility['error']}")
        if not eligibility["eligible"]:
            reasons = " ".join(eligibility["reasons"])
            return AIMessage(content=f"I'm sorry, order {order_id} can't be refunded. {reasons}")

        refund = turn.last(ISSUE_REFUND)
        if refund is None:
            return _call(
                ISSUE_REFUND,
                order_id=order_id,
                amount=turn.amount or eligibility["refundable_amount"],
                reason=f"Customer request: {turn.request}"[:500],
            )
        if "error" in refund:
            return AIMessage(content=f"I wasn't able to issue the refund. {refund['error']}")
        return AIMessage(
            content=(
                f"Done! I've refunded {refund['amount']} USD for order {order_id} "
                f"(refund {refund['refund_id']}). It goes back to your original payment method "
                "and usually appears within 5 to 10 business days."
            )
        )

    def _order_status(self, turn: _Turn) -> AIMessage:
        order_id = turn.order_id
        if order_id is None:
            return self._ask_for_order(turn, purpose="looked up")
        order = turn.last(LOOKUP_ORDER)
        if order is None:
            return _call(LOOKUP_ORDER, order_id=order_id)
        if "error" in order:
            return AIMessage(
                content=f"I couldn't find that order: {order['error']} Please check the number."
            )
        if turn.wants_refund and not turn.transferred:
            return _call(TRANSFER_TOOL, target="billing", reason="The customer wants a refund.")
        wants_cancel = bool(_matches(turn.request, ("cancel",)))
        return AIMessage(content=_describe_order(order, wants_cancel=wants_cancel))

    def _technical(self, turn: _Turn) -> AIMessage:
        if turn.wants_refund and not turn.transferred:
            return _call(TRANSFER_TOOL, target="billing", reason="The customer wants a refund.")
        return self._answer_from_kb(turn, ticket_category="technical")

    @staticmethod
    def _escalation(turn: _Turn) -> AIMessage:
        ticket = turn.last(CREATE_SUPPORT_TICKET)
        if ticket is None:
            args: dict[str, Any] = {
                "subject": _subject(turn.request),
                "description": f"Customer asked for escalation: {turn.request}"[:2000],
                "category": "escalation",
                "priority": "high",
            }
            if turn.order_id is not None:
                args["order_id"] = turn.order_id
            return _call(CREATE_SUPPORT_TICKET, **args)
        if "error" in ticket:
            return AIMessage(content=f"I wasn't able to open a ticket: {ticket['error']}")
        return AIMessage(
            content=(
                "I'm sorry about this. I've escalated your request to our support team as "
                f"ticket {ticket['ticket_id']} (high priority). A member of the team will "
                "follow up."
            )
        )

    # -- shared steps ---------------------------------------------------------------------

    @staticmethod
    def _ask_for_order(turn: _Turn, *, purpose: str) -> AIMessage:
        listing = turn.last(LIST_CUSTOMER_ORDERS)
        if turn.customer_id is not None and listing is None:
            return _call(LIST_CUSTOMER_ORDERS)
        if listing is not None and listing.get("orders"):
            orders = "; ".join(
                f"{o['order_id']} ({', '.join(o['items'])}, {o['status']})"
                for o in listing["orders"]
            )
            return AIMessage(
                content=f"Which order should be {purpose}? Your recent orders: {orders}."
            )
        return AIMessage(
            content=f"Could you tell me the number of the order to be {purpose}? "
            "It looks like ORD-12345."
        )

    @staticmethod
    def _answer_from_kb(turn: _Turn, *, ticket_category: str | None) -> AIMessage:
        search = turn.last(SEARCH_KNOWLEDGE_BASE)
        if search is None:
            query = turn.request[:300] if len(turn.request) >= 2 else "help"
            return _call(SEARCH_KNOWLEDGE_BASE, query=query, k=2)
        results = search.get("results") or []
        if results:
            top = results[0]
            return AIMessage(
                content=f'Here is what our help center says in "{top["title"]}": {top["snippet"]}'
            )
        if ticket_category is None:
            return AIMessage(
                content="I couldn't find an answer to that in our help center. "
                "Could you share a few more details?"
            )
        ticket = turn.last(CREATE_SUPPORT_TICKET)
        if ticket is None:
            return _call(
                CREATE_SUPPORT_TICKET,
                subject=_subject(turn.request),
                description=f"Customer wrote: {turn.request}"[:2000],
                category=ticket_category,
            )
        if "error" in ticket:
            return AIMessage(content=f"I wasn't able to open a ticket: {ticket['error']}")
        return AIMessage(
            content="I couldn't find an answer in our help center, so I've opened ticket "
            f"{ticket['ticket_id']} for our support team."
        )
