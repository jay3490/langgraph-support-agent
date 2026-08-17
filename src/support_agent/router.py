"""The router node: classify the latest customer message with structured output."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from support_agent.prompts import ROUTER_PROMPT
from support_agent.state import CATEGORY_TO_SPECIALIST, Category, SupportState

logger = logging.getLogger(__name__)

FALLBACK_REASON = "The request could not be classified automatically."


class RouteDecision(BaseModel):
    """Route the customer's latest message to the team that should handle it."""

    category: Category = Field(description="The single best category for the latest message.")
    reason: str = Field(description="One short sentence explaining the choice.", max_length=300)


def router_context(messages: Sequence[BaseMessage], limit: int = 6) -> list[BaseMessage]:
    """The recent conversation without tool traffic, ending with the latest message.

    Tool calls and tool results are dropped: the router only needs what was said, and
    an assistant message with tool calls but without its tool results is invalid input
    for most chat APIs.
    """
    conversation = [
        m
        for m in messages
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return conversation[-limit:]


RouterNode = Callable[
    [SupportState],
    Awaitable[Command[Literal["billing", "technical", "order_status", "escalation"]]],
]


def make_router(model: BaseChatModel) -> RouterNode:
    classifier = model.with_structured_output(
        RouteDecision, method="function_calling", include_raw=True
    )

    async def router(
        state: SupportState,
    ) -> Command[Literal["billing", "technical", "order_status", "escalation"]]:
        prompt = [SystemMessage(ROUTER_PROMPT), *router_context(state["messages"])]
        result = await classifier.ainvoke(prompt)
        decision = result["parsed"]
        if not isinstance(decision, RouteDecision):
            logger.warning("Router output could not be parsed: %s", result.get("parsing_error"))
            decision = RouteDecision(category="escalate", reason=FALLBACK_REASON)
        specialist = CATEGORY_TO_SPECIALIST[decision.category]
        logger.info("Routed to %s: %s", specialist, decision.reason)
        return Command(
            goto=specialist,
            update={
                "route": decision.category,
                "route_reason": decision.reason,
                "active_agent": specialist,
                "visited": [specialist],
                "handoffs": 0,
                "tool_calls_this_turn": 0,
                "step_limit_hit": False,
                "refund_decisions": {},
                "final_response": None,
            },
        )

    return router
