from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from support_agent.db import connect
from support_agent.money import format_cents, to_cents
from support_agent.repository import (
    ApprovalRequiredError,
    NotFoundError,
    RefundError,
    RefundPolicy,
    SupportRepository,
)
from tests.conftest import TODAY


def test_money_round_trip() -> None:
    assert to_cents(Decimal("89.99")) == 8999
    assert to_cents("0.005") == 1
    assert to_cents(12) == 1200
    assert format_cents(34900) == "349.00"


def test_get_order_includes_items_and_refunds(repository: SupportRepository) -> None:
    order = repository.get_order("ORD-10009")
    assert order.customer_id == "CUST-1002"
    assert order.total_cents == 12900
    assert order.refunded_cents == 3000
    payload = order.to_dict()
    assert payload["total"] == "129.00"
    assert payload["items"][0]["name"] == "Mechanical keyboard"


def test_get_order_unknown(repository: SupportRepository) -> None:
    with pytest.raises(NotFoundError, match="ORD-99999"):
        repository.get_order("ORD-99999")


def test_list_orders_newest_first(repository: SupportRepository) -> None:
    orders = repository.list_orders("CUST-1001")
    assert [o.id for o in orders] == ["ORD-10003", "ORD-10001", "ORD-10002"]
    assert [o.id for o in repository.list_orders("CUST-1001", limit=1)] == ["ORD-10003"]


def test_list_orders_unknown_customer(repository: SupportRepository) -> None:
    with pytest.raises(NotFoundError):
        repository.list_orders("CUST-9999")


def test_get_customer(repository: SupportRepository) -> None:
    assert repository.get_customer("CUST-1002").tier == "gold"


# -- refund eligibility rules -------------------------------------------------------------


def test_recent_delivery_below_threshold_is_eligible(repository: SupportRepository) -> None:
    result = repository.check_refund_eligibility("ORD-10001")
    assert result.eligible
    assert result.refundable_cents == 8999
    assert not result.requires_approval


def test_amount_above_threshold_requires_approval(repository: SupportRepository) -> None:
    result = repository.check_refund_eligibility("ORD-10002")
    assert result.eligible
    assert result.requires_approval
    partial = repository.check_refund_eligibility("ORD-10002", amount_cents=5000)
    assert partial.eligible and not partial.requires_approval


def test_threshold_is_exclusive(db_path: Path) -> None:
    repo = SupportRepository(
        db_path, policy=RefundPolicy(approval_threshold_cents=8999), today=lambda: TODAY
    )
    assert not repo.check_refund_eligibility("ORD-10001").requires_approval
    repo = SupportRepository(
        db_path, policy=RefundPolicy(approval_threshold_cents=8998), today=lambda: TODAY
    )
    assert repo.check_refund_eligibility("ORD-10001").requires_approval


def test_window_boundary_is_inclusive(repository: SupportRepository, db_path: Path) -> None:
    on_edge = repository.check_refund_eligibility("ORD-10010")  # delivered 30 days ago
    assert on_edge.eligible
    assert on_edge.window_ends == TODAY

    tomorrow = SupportRepository(db_path, today=lambda: TODAY + timedelta(days=1))
    past_edge = tomorrow.check_refund_eligibility("ORD-10010")
    assert not past_edge.eligible
    assert "refund window ended" in past_edge.reasons[0]


def test_gold_tier_gets_longer_window(repository: SupportRepository) -> None:
    gold = repository.check_refund_eligibility("ORD-10004")  # gold, 40 days
    standard = repository.check_refund_eligibility("ORD-10005")  # standard, 40 days
    assert gold.eligible
    assert gold.window_ends == TODAY + timedelta(days=5)
    assert not standard.eligible


@pytest.mark.parametrize(
    ("order_id", "status"),
    [("ORD-10003", "shipped"), ("ORD-10006", "processing"), ("ORD-10008", "cancelled")],
)
def test_only_delivered_orders_are_refundable(
    repository: SupportRepository, order_id: str, status: str
) -> None:
    result = repository.check_refund_eligibility(order_id)
    assert not result.eligible
    assert f"'{status}'" in result.reasons[0]
    assert result.refundable_cents == 0


def test_digital_and_gift_card_items_are_excluded(repository: SupportRepository) -> None:
    result = repository.check_refund_eligibility("ORD-10007")
    assert result.eligible
    assert result.refundable_cents == 3900


def test_previous_refunds_are_subtracted(repository: SupportRepository) -> None:
    result = repository.check_refund_eligibility("ORD-10009")
    assert result.refundable_cents == 9900
    assert result.already_refunded_cents == 3000
    too_much = repository.check_refund_eligibility("ORD-10009", amount_cents=9901)
    assert not too_much.eligible
    assert "exceeds the refundable amount 99.00" in too_much.reasons[0]


def test_fully_refunded_order(repository: SupportRepository) -> None:
    repository.issue_refund("ORD-10001", 8999, "Damaged")
    result = repository.check_refund_eligibility("ORD-10001")
    assert not result.eligible
    assert "already been refunded in full" in result.reasons[0]


def test_order_with_only_non_refundable_items(db_path: Path, repository: SupportRepository) -> None:
    conn = connect(db_path)
    with conn:
        conn.execute("DELETE FROM order_items WHERE order_id = 'ORD-10007' AND sku = 'ACC-410'")
    conn.close()
    result = repository.check_refund_eligibility("ORD-10007")
    assert not result.eligible
    assert "None of the items" in result.reasons[0]


# -- issuing refunds ----------------------------------------------------------------------


def test_issue_refund_below_threshold(repository: SupportRepository) -> None:
    refund = repository.issue_refund("ORD-10001", 8999, "Arrived damaged")
    assert refund.id == "REF-000002"
    assert refund.approved_by == "auto"
    assert repository.get_order("ORD-10001").refunded_cents == 8999


def test_issue_refund_above_threshold_needs_reviewer(repository: SupportRepository) -> None:
    with pytest.raises(ApprovalRequiredError):
        repository.issue_refund("ORD-10002", 34900, "Wobbly")
    assert repository.get_order("ORD-10002").refunded_cents == 0

    refund = repository.issue_refund("ORD-10002", 34900, "Wobbly", approved_by="sam")
    assert refund.approved_by == "sam"


def test_issue_refund_rechecks_rules(repository: SupportRepository) -> None:
    with pytest.raises(RefundError, match="only delivered orders"):
        repository.issue_refund("ORD-10003", 1000, "Changed my mind")
    with pytest.raises(RefundError, match="exceeds"):
        repository.issue_refund("ORD-10009", 10000, "Too much")
    with pytest.raises(RefundError, match="positive"):
        repository.issue_refund("ORD-10001", 0, "Nothing")


def test_eligibility_uses_injected_clock(db_path: Path) -> None:
    far_future = SupportRepository(db_path, today=lambda: date(2030, 1, 1))
    assert not far_future.check_refund_eligibility("ORD-10001").eligible
