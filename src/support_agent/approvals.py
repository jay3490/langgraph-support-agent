"""Human-in-the-loop review of large refunds."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.messages import ToolCall
from pydantic import BaseModel, Field, ValidationError

from support_agent.money import format_cents, to_cents
from support_agent.tools import ISSUE_REFUND, IssueRefundArgs


class ReviewDecision(BaseModel):
    """What a human reviewer decided about the refunds a turn is waiting on."""

    approved: bool
    reviewer: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=500)


@dataclass(frozen=True)
class PendingRefund:
    tool_call_id: str
    order_id: str
    amount: Decimal
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_call_id": self.tool_call_id,
            "order_id": self.order_id,
            "amount": format_cents(to_cents(self.amount)),
            "reason": self.reason,
        }


def refunds_needing_review(
    calls: Iterable[ToolCall],
    threshold_cents: int,
    decisions: Mapping[str, Any],
) -> list[PendingRefund]:
    """Refund calls above the threshold that no reviewer has decided on yet.

    Calls with invalid arguments are left out: the tools node rejects them with a
    validation error, so there is nothing for a human to review.
    """
    pending: list[PendingRefund] = []
    for call in calls:
        if call["name"] != ISSUE_REFUND or not call["id"] or call["id"] in decisions:
            continue
        try:
            args = IssueRefundArgs.model_validate(call["args"])
        except ValidationError:
            continue
        if to_cents(args.amount) > threshold_cents:
            pending.append(PendingRefund(call["id"], args.order_id, args.amount, args.reason))
    return pending


def approval_request(
    pending: list[PendingRefund], *, threshold_cents: int, customer_id: str | None
) -> dict[str, Any]:
    """The payload surfaced to the reviewer through ``interrupt``."""
    return {
        "type": "refund_approval",
        "customer_id": customer_id,
        "threshold": format_cents(threshold_cents),
        "refunds": [refund.to_dict() for refund in pending],
    }
