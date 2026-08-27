"""The LangGraph workflow: router, specialists, supervisor, tools and final response.

::

    START -> router -> <specialist> -> supervisor -> tools -> <active specialist> ...
                                                 \\-> approval -> tools    (large refunds)
                                                 \\-> <other specialist>   (handoff)
                                                 \\-> respond -> END

Every specialist reply goes to the supervisor, which is the single place where the
per-turn guardrails are enforced: the tool-call budget, the handoff policy and the
human review of refunds above the approval threshold. The approval node pauses the
graph with ``interrupt``; with a checkpointer the paused thread survives restarts.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from support_agent.approvals import ReviewDecision, approval_request, refunds_needing_review
from support_agent.money import format_cents
from support_agent.prompts import specialist_prompt
from support_agent.repository import Ticket
from support_agent.router import make_router
from support_agent.state import SPECIALISTS, Specialist, SupportState
from support_agent.tools import (
    CHECK_REFUND_ELIGIBILITY,
    CREATE_SUPPORT_TICKET,
    ISSUE_REFUND,
    LIST_CUSTOMER_ORDERS,
    LOOKUP_ORDER,
    SEARCH_KNOWLEDGE_BASE,
    ToolContext,
    Toolkit,
    tool_message,
)

logger = logging.getLogger(__name__)

TRANSFER_TOOL = "transfer_to_specialist"

AGENT_TOOLS: dict[Specialist, tuple[str, ...]] = {
    "billing": (
        LOOKUP_ORDER,
        LIST_CUSTOMER_ORDERS,
        CHECK_REFUND_ELIGIBILITY,
        ISSUE_REFUND,
        SEARCH_KNOWLEDGE_BASE,
    ),
    "technical": (SEARCH_KNOWLEDGE_BASE, CREATE_SUPPORT_TICKET),
    "order_status": (
        LOOKUP_ORDER,
        LIST_CUSTOMER_ORDERS,
        SEARCH_KNOWLEDGE_BASE,
        CREATE_SUPPORT_TICKET,
    ),
    "escalation": (LOOKUP_ORDER, LIST_CUSTOMER_ORDERS, CREATE_SUPPORT_TICKET),
}

FALLBACK_REPLY = "Sorry, I could not put together an answer. Could you rephrase your request?"


class TransferArgs(BaseModel):
    target: Specialist = Field(description="The specialist who should take over.")
    reason: str = Field(min_length=3, max_length=300, description="Why they are better placed.")


def _transfer_schema() -> dict[str, Any]:
    spec = convert_to_openai_tool(TransferArgs)
    spec["function"]["name"] = TRANSFER_TOOL
    spec["function"]["description"] = (
        "Hand the conversation to another specialist who is better placed to help."
    )
    return spec


def _last_ai_message(state: SupportState) -> AIMessage | None:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message
    return None


def _last_customer_message(state: SupportState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


class SupportWorkflow:
    """Builds the support graph around one chat model and one toolkit."""

    def __init__(
        self,
        model: BaseChatModel,
        toolkit: Toolkit,
        *,
        max_tool_steps: int = 8,
        max_handoffs: int = 2,
    ) -> None:
        self.model = model
        self.toolkit = toolkit
        self.max_tool_steps = max_tool_steps
        self.max_handoffs = max_handoffs
        self._transfer = _transfer_schema()

    @property
    def threshold_cents(self) -> int:
        return self.toolkit.repository.policy.approval_threshold_cents

    # -- wiring ---------------------------------------------------------------------------

    def build(self) -> StateGraph:
        graph = StateGraph(SupportState)
        graph.add_node("router", make_router(self.model))
        for name in SPECIALISTS:
            graph.add_node(name, self._specialist(name))
            graph.add_edge(name, "supervisor")
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("approval", self.approval)
        graph.add_node("tools", self.tools)
        graph.add_node("respond", self.respond)

        graph.add_edge(START, "router")
        graph.add_conditional_edges("tools", self._back_to_active_agent, list(SPECIALISTS))
        graph.add_edge("respond", END)
        return graph

    def compile(self, checkpointer: BaseCheckpointSaver[Any] | None = None) -> CompiledStateGraph:
        return self.build().compile(checkpointer=checkpointer)

    @staticmethod
    def _back_to_active_agent(state: SupportState) -> Specialist:
        agent = state.get("active_agent")
        if agent is None:
            raise RuntimeError("Tools ran without an active specialist.")
        return agent

    # -- nodes ----------------------------------------------------------------------------

    def _specialist(self, name: Specialist) -> Callable[[SupportState], Awaitable[dict[str, Any]]]:
        bound = self.model.bind_tools([*self.toolkit.schemas(AGENT_TOOLS[name]), self._transfer])

        async def specialist(state: SupportState) -> dict[str, Any]:
            system = SystemMessage(
                specialist_prompt(
                    name,
                    customer_id=state.get("customer_id"),
                    threshold=format_cents(self.threshold_cents),
                )
            )
            reply = await bound.ainvoke([system, *state["messages"]])
            return {"messages": [reply.model_copy(update={"name": name})]}

        specialist.__name__ = name
        return specialist

    async def supervisor(
        self, state: SupportState
    ) -> Command[
        Literal[
            "tools", "approval", "respond", "billing", "technical", "order_status", "escalation"
        ]
    ]:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        if not calls:
            return Command(goto="respond")

        used = state.get("tool_calls_this_turn", 0)
        if used + len(calls) > self.max_tool_steps:
            logger.warning("Tool-call budget of %d reached; ending the turn", self.max_tool_steps)
            error = (
                f"Not executed: the limit of {self.max_tool_steps} tool calls per turn was reached."
            )
            return Command(
                goto="respond",
                update={
                    "messages": [tool_message(c, {"error": error}, error=True) for c in calls],
                    "step_limit_hit": True,
                },
            )

        transfers = [c for c in calls if c["name"] == TRANSFER_TOOL]
        if transfers:
            return self._handoff(state, calls, transfers[0])
        if refunds_needing_review(calls, self.threshold_cents, state.get("refund_decisions", {})):
            return Command(goto="approval")
        return Command(goto="tools")

    def _handoff(
        self, state: SupportState, calls: list[ToolCall], transfer: ToolCall
    ) -> Command[Any]:
        agent = state.get("active_agent")
        visited = state.get("visited", [])
        handoffs = state.get("handoffs", 0)
        others = [c for c in calls if c is not transfer]

        refusal: str | None = None
        target: Specialist | None = None
        try:
            args = TransferArgs.model_validate(transfer["args"])
        except ValidationError as exc:
            refusal = f"Invalid transfer: {exc.errors(include_url=False)[0]['msg']}"
        else:
            target = args.target
            if target == agent:
                refusal = f"You are the {agent} specialist already; answer the customer yourself."
            elif target in visited:
                refusal = (
                    f"The {target} specialist already handled this turn; "
                    "answer the customer yourself."
                )
            elif handoffs >= self.max_handoffs:
                refusal = "No more handoffs are allowed this turn; answer the customer yourself."

        if refusal is not None or target is None:
            logger.info("Handoff from %s refused: %s", agent, refusal)
            skipped = "Not executed: the transfer request in the same message was refused."
            return Command(
                goto=agent,
                update={
                    "messages": [
                        tool_message(transfer, {"error": refusal}, error=True),
                        *(tool_message(c, {"error": skipped}, error=True) for c in others),
                    ],
                    "tool_calls_this_turn": state.get("tool_calls_this_turn", 0) + len(calls),
                },
            )

        logger.info("Handoff %s -> %s: %s", agent, target, args.reason)
        skipped = f"Not executed: the conversation was transferred to {target}."
        return Command(
            goto=target,
            update={
                "messages": [
                    tool_message(transfer, {"transferred_to": target, "reason": args.reason}),
                    *(tool_message(c, {"error": skipped}, error=True) for c in others),
                ],
                "active_agent": target,
                "visited": [*visited, target],
                "handoffs": handoffs + 1,
            },
        )

    async def approval(self, state: SupportState) -> Command[Literal["tools"]]:
        """Pause until a human approves or rejects the large refunds in the last reply.

        Nothing before ``interrupt`` has side effects, because LangGraph re-runs the
        whole node when the thread is resumed.
        """
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        decisions = dict(state.get("refund_decisions", {}))
        pending = refunds_needing_review(calls, self.threshold_cents, decisions)
        if not pending:
            return Command(goto="tools")

        request = approval_request(
            pending, threshold_cents=self.threshold_cents, customer_id=state.get("customer_id")
        )
        decision = ReviewDecision.model_validate(interrupt(request))
        logger.info(
            "Reviewer %s %s %d refund(s)",
            decision.reviewer,
            "approved" if decision.approved else "rejected",
            len(pending),
        )
        decisions.update({refund.tool_call_id: decision.model_dump() for refund in pending})
        return Command(goto="tools", update={"refund_decisions": decisions})

    async def tools(self, state: SupportState) -> dict[str, Any]:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        decisions = state.get("refund_decisions", {})
        customer_id = state.get("customer_id")
        results: list[ToolMessage] = []
        for call in calls:
            decision = decisions.get(call["id"] or "") if call["name"] == ISSUE_REFUND else None
            if decision is not None and not decision["approved"]:
                note = f": {decision['note']}" if decision.get("note") else "."
                error = f"The refund was rejected by reviewer {decision['reviewer']}{note}"
                results.append(tool_message(call, {"error": error}, error=True))
                continue
            context = ToolContext(
                customer_id=customer_id,
                reviewer=decision["reviewer"] if decision is not None else None,
            )
            results.append(await asyncio.to_thread(self.toolkit.execute, call, context))
        return {
            "messages": results,
            "tool_calls_this_turn": state.get("tool_calls_this_turn", 0) + len(calls),
        }

    async def respond(self, state: SupportState) -> dict[str, Any]:
        if state.get("step_limit_hit"):
            ticket = await asyncio.to_thread(self._step_limit_ticket, state)
            text = (
                "I'm sorry, I couldn't finish this request automatically. I've opened ticket "
                f"{ticket.id} so a member of our support team can follow up."
            )
            return {"messages": [AIMessage(content=text, name="support")], "final_response": text}

        last = _last_ai_message(state)
        text = last.text.strip() if last is not None else ""
        return {"final_response": text or FALLBACK_REPLY}

    def _step_limit_ticket(self, state: SupportState) -> Ticket:
        request = _last_customer_message(state)
        return self.toolkit.repository.create_ticket(
            subject="Assistant could not complete the request",
            description=(
                f"The automated assistant ({state.get('active_agent')}) reached its limit of "
                f"{self.max_tool_steps} tool calls. Customer message: {request}"
            ),
            category="escalation",
            priority="high",
            customer_id=state.get("customer_id"),
        )
