"""Graph state shared by every node, and the names of the specialist agents."""

from __future__ import annotations

from typing import Any, Literal, NotRequired

from langgraph.graph import MessagesState

Specialist = Literal["billing", "technical", "order_status", "escalation"]
SPECIALISTS: tuple[Specialist, ...] = ("billing", "technical", "order_status", "escalation")

Category = Literal["billing", "technical", "order_status", "escalate"]
CATEGORY_TO_SPECIALIST: dict[Category, Specialist] = {
    "billing": "billing",
    "technical": "technical",
    "order_status": "order_status",
    "escalate": "escalation",
}


class SupportState(MessagesState):
    """State of one support thread.

    ``messages`` accumulates over the whole thread; the remaining keys describe the
    current turn and are reset by the router when a new customer message arrives.
    """

    customer_id: NotRequired[str | None]
    route: NotRequired[Category | None]
    route_reason: NotRequired[str]
    active_agent: NotRequired[Specialist | None]
    visited: NotRequired[list[Specialist]]
    handoffs: NotRequired[int]
    tool_calls_this_turn: NotRequired[int]
    step_limit_hit: NotRequired[bool]
    # Reviewer decisions for refund tool calls, keyed by tool-call id.
    refund_decisions: NotRequired[dict[str, dict[str, Any]]]
    final_response: NotRequired[str | None]
