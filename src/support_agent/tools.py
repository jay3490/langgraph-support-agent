"""The tools the specialist agents can call, with pydantic-validated arguments.

Tools are executed by the graph, not by LangChain's tool runtime, so that every call
goes through the same path: validate the arguments against the tool's pydantic model,
apply conversation scoping (a thread opened for a customer can only see that customer's
orders), run the handler and turn the outcome into a :class:`ToolMessage`. Validation
and business-rule errors are returned to the model as error messages instead of
raising, so it can correct itself.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
)

from support_agent.kb import KnowledgeBase
from support_agent.money import to_cents
from support_agent.repository import (
    AUTO_APPROVER,
    NotFoundError,
    SupportDataError,
    SupportRepository,
    TicketCategory,
    TicketPriority,
)

logger = logging.getLogger(__name__)

LOOKUP_ORDER = "lookup_order"
LIST_CUSTOMER_ORDERS = "list_customer_orders"
CHECK_REFUND_ELIGIBILITY = "check_refund_eligibility"
ISSUE_REFUND = "issue_refund"
SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
CREATE_SUPPORT_TICKET = "create_support_ticket"


def _normalize_id(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


# Field comes before the validator so the pattern also lands in the JSON schema.
OrderId = Annotated[
    str,
    Field(pattern=r"^ORD-\d{5}$", description="Order number, for example ORD-10001."),
    BeforeValidator(_normalize_id),
]
CustomerId = Annotated[
    str,
    Field(pattern=r"^CUST-\d{4}$", description="Customer id, for example CUST-1001."),
    BeforeValidator(_normalize_id),
]
Amount = Annotated[
    Decimal,
    Field(gt=0, max_digits=10, decimal_places=2),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0, "description": "Amount in USD."}),
]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LookupOrderArgs(_Args):
    order_id: OrderId


class ListCustomerOrdersArgs(_Args):
    customer_id: CustomerId | None = Field(
        default=None, description="Defaults to the customer of this conversation."
    )
    limit: int = Field(default=5, ge=1, le=20)


class RefundEligibilityArgs(_Args):
    order_id: OrderId
    amount: Amount | None = Field(
        default=None, description="Requested amount; omit to check the full refundable amount."
    )


class IssueRefundArgs(_Args):
    order_id: OrderId
    amount: Amount
    reason: str = Field(min_length=3, max_length=500)


class SearchKnowledgeBaseArgs(_Args):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=3, ge=1, le=5, description="Number of articles to return.")


class CreateTicketArgs(_Args):
    subject: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=10, max_length=2000)
    category: TicketCategory
    priority: TicketPriority = "normal"
    order_id: OrderId | None = None
    customer_id: CustomerId | None = Field(
        default=None, description="Defaults to the customer of this conversation."
    )


class AccessDeniedError(SupportDataError):
    pass


@dataclass(frozen=True)
class ToolContext:
    """Per-call information that does not come from the model.

    ``customer_id`` is the customer the conversation was opened for (if any);
    ``reviewer`` is the human who approved this specific call (refunds only).
    """

    customer_id: str | None = None
    reviewer: str | None = None


Handler = Callable[[Any, ToolContext], dict[str, Any]]


@dataclass(frozen=True)
class SupportTool:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: Handler

    def schema(self) -> dict[str, Any]:
        """The tool in OpenAI function-calling format, ready for ``bind_tools``."""
        spec = convert_to_openai_tool(self.args_schema)
        spec["function"]["name"] = self.name
        spec["function"]["description"] = self.description
        return spec


def tool_message(call: ToolCall, payload: dict[str, Any], *, error: bool = False) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id=call["id"] or "",
        name=call["name"],
        status="error" if error else "success",
    )


def _format_validation_error(name: str, exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'arguments'}: {err['msg']}"
        for err in exc.errors(include_url=False)
    )
    return f"Invalid arguments for {name}: {problems}"


class Toolkit:
    """All support tools bound to one repository and knowledge base."""

    def __init__(self, repository: SupportRepository, kb: KnowledgeBase) -> None:
        self.repository = repository
        self.kb = kb
        tools = [
            SupportTool(
                LOOKUP_ORDER,
                "Look up one order: status, dates, carrier, tracking number, items and refunds.",
                LookupOrderArgs,
                self._lookup_order,
            ),
            SupportTool(
                LIST_CUSTOMER_ORDERS,
                "List a customer's most recent orders, newest first.",
                ListCustomerOrdersArgs,
                self._list_customer_orders,
            ),
            SupportTool(
                CHECK_REFUND_ELIGIBILITY,
                "Check whether an order can be refunded under the refund policy, how much is"
                " refundable and whether the refund needs human approval.",
                RefundEligibilityArgs,
                self._check_refund_eligibility,
            ),
            SupportTool(
                ISSUE_REFUND,
                "Issue a refund to the original payment method. Check eligibility first."
                " Refunds above the approval threshold are paused until a human approves.",
                IssueRefundArgs,
                self._issue_refund,
            ),
            SupportTool(
                SEARCH_KNOWLEDGE_BASE,
                "Search the help-center articles and return the best matching passages.",
                SearchKnowledgeBaseArgs,
                self._search_knowledge_base,
            ),
            SupportTool(
                CREATE_SUPPORT_TICKET,
                "Open a support ticket for the human support team.",
                CreateTicketArgs,
                self._create_ticket,
            ),
        ]
        self.tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> SupportTool:
        return self.tools[name]

    def schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        return [self.tools[name].schema() for name in names]

    def execute(self, call: ToolCall, context: ToolContext) -> ToolMessage:
        name = call["name"]
        tool = self.tools.get(name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", name)
            return tool_message(call, {"error": f"Unknown tool: {name}"}, error=True)
        try:
            args = tool.args_schema.model_validate(call["args"])
        except ValidationError as exc:
            message = _format_validation_error(name, exc)
            logger.info("Rejected %s call: %s", name, message)
            return tool_message(call, {"error": message}, error=True)

        logger.info("Calling %s with %s", name, args.model_dump(mode="json"))
        try:
            result = tool.handler(args, context)
        except SupportDataError as exc:
            logger.info("%s failed: %s", name, exc)
            return tool_message(call, {"error": str(exc)}, error=True)
        return tool_message(call, result)

    # -- handlers ---------------------------------------------------------------------

    def _owned_order(self, order_id: str, context: ToolContext) -> None:
        order = self.repository.get_order(order_id)
        if context.customer_id is not None and order.customer_id != context.customer_id:
            # Same message as a missing order, so ids of other customers do not leak.
            raise NotFoundError(f"Order {order_id} was not found.")

    def _lookup_order(self, args: LookupOrderArgs, context: ToolContext) -> dict[str, Any]:
        self._owned_order(args.order_id, context)
        return self.repository.get_order(args.order_id).to_dict()

    def _list_customer_orders(
        self, args: ListCustomerOrdersArgs, context: ToolContext
    ) -> dict[str, Any]:
        customer_id = args.customer_id or context.customer_id
        if customer_id is None:
            raise SupportDataError(
                "No customer is linked to this conversation; ask the customer for an order number."
            )
        if context.customer_id is not None and customer_id != context.customer_id:
            raise AccessDeniedError("Only the orders of this conversation's customer are visible.")
        orders = self.repository.list_orders(customer_id, limit=args.limit)
        return {
            "customer_id": customer_id,
            "orders": [
                {
                    "order_id": o.id,
                    "status": o.status,
                    "placed_on": o.placed_on.isoformat(),
                    "total": o.to_dict()["total"],
                    "items": [item.name for item in o.items],
                }
                for o in orders
            ],
        }

    def _check_refund_eligibility(
        self, args: RefundEligibilityArgs, context: ToolContext
    ) -> dict[str, Any]:
        self._owned_order(args.order_id, context)
        amount = to_cents(args.amount) if args.amount is not None else None
        return self.repository.check_refund_eligibility(args.order_id, amount).to_dict()

    def _issue_refund(self, args: IssueRefundArgs, context: ToolContext) -> dict[str, Any]:
        self._owned_order(args.order_id, context)
        refund = self.repository.issue_refund(
            args.order_id,
            to_cents(args.amount),
            args.reason,
            approved_by=context.reviewer or AUTO_APPROVER,
        )
        return refund.to_dict()

    def _search_knowledge_base(
        self, args: SearchKnowledgeBaseArgs, context: ToolContext
    ) -> dict[str, Any]:
        hits = self.kb.search(args.query, k=args.k)
        return {"query": args.query, "results": [hit.to_dict() for hit in hits]}

    def _create_ticket(self, args: CreateTicketArgs, context: ToolContext) -> dict[str, Any]:
        if args.order_id is not None:
            self._owned_order(args.order_id, context)
        customer_id = context.customer_id or args.customer_id
        ticket = self.repository.create_ticket(
            subject=args.subject,
            description=args.description,
            category=args.category,
            priority=args.priority,
            customer_id=customer_id,
            order_id=args.order_id,
        )
        return ticket.to_dict()
