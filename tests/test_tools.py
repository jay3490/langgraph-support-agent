from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import ToolCall, ToolMessage

from support_agent.repository import NotFoundError, SupportRepository
from support_agent.tools import (
    CHECK_REFUND_ELIGIBILITY,
    CREATE_SUPPORT_TICKET,
    ISSUE_REFUND,
    LIST_CUSTOMER_ORDERS,
    LOOKUP_ORDER,
    SEARCH_KNOWLEDGE_BASE,
    ToolContext,
    Toolkit,
)

ALICE = ToolContext(customer_id="CUST-1001")
NOBODY = ToolContext()


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, args=args, id=f"call-{name}", type="tool_call")


def run(toolkit: Toolkit, name: str, context: ToolContext = NOBODY, **args: Any) -> ToolMessage:
    message = toolkit.execute(call(name, **args), context)
    assert message.tool_call_id == f"call-{name}"
    assert message.name == name
    return message


def payload(message: ToolMessage) -> dict[str, Any]:
    return json.loads(str(message.content))


def test_schemas_use_tool_names_and_pydantic_constraints(toolkit: Toolkit) -> None:
    schema = toolkit.get(LOOKUP_ORDER).schema()
    function = schema["function"]
    assert function["name"] == "lookup_order"
    assert function["parameters"]["properties"]["order_id"]["pattern"] == r"^ORD-\d{5}$"
    assert function["parameters"]["required"] == ["order_id"]
    assert len(toolkit.schemas([LOOKUP_ORDER, ISSUE_REFUND])) == 2


# -- lookup_order ----------------------------------------------------------------------


def test_lookup_order(toolkit: Toolkit) -> None:
    message = run(toolkit, LOOKUP_ORDER, order_id="ORD-10003")
    assert message.status == "success"
    data = payload(message)
    assert data["status"] == "shipped"
    assert data["tracking_number"] == "1Z999AA10000000003"


def test_order_ids_are_normalized(toolkit: Toolkit) -> None:
    assert payload(run(toolkit, LOOKUP_ORDER, order_id="  ord-10001 "))["order_id"] == "ORD-10001"


@pytest.mark.parametrize(
    "args",
    [{"order_id": "10001"}, {"order_id": "ORD-1"}, {}, {"order_id": "ORD-10001", "extra": 1}],
)
def test_invalid_arguments_become_error_messages(toolkit: Toolkit, args: dict[str, Any]) -> None:
    message = toolkit.execute(call(LOOKUP_ORDER, **args), NOBODY)
    assert message.status == "error"
    assert payload(message)["error"].startswith("Invalid arguments for lookup_order")


def test_unknown_order_and_unknown_tool(toolkit: Toolkit) -> None:
    missing = run(toolkit, LOOKUP_ORDER, order_id="ORD-99999")
    assert missing.status == "error"
    assert "was not found" in payload(missing)["error"]

    unknown = toolkit.execute(call("drop_tables"), NOBODY)
    assert unknown.status == "error"
    assert payload(unknown) == {"error": "Unknown tool: drop_tables"}


def test_orders_of_other_customers_look_missing(toolkit: Toolkit) -> None:
    own = run(toolkit, LOOKUP_ORDER, ALICE, order_id="ORD-10001")
    other = run(toolkit, LOOKUP_ORDER, ALICE, order_id="ORD-10004")
    assert own.status == "success"
    assert other.status == "error"
    assert payload(other)["error"] == "Order ORD-10004 was not found."


# -- list_customer_orders --------------------------------------------------------------


def test_list_orders_defaults_to_conversation_customer(toolkit: Toolkit) -> None:
    data = payload(run(toolkit, LIST_CUSTOMER_ORDERS, ALICE))
    assert data["customer_id"] == "CUST-1001"
    assert [o["order_id"] for o in data["orders"]] == ["ORD-10003", "ORD-10001", "ORD-10002"]


def test_list_orders_needs_a_customer(toolkit: Toolkit) -> None:
    message = run(toolkit, LIST_CUSTOMER_ORDERS)
    assert message.status == "error"
    assert "order number" in payload(message)["error"]


def test_list_orders_is_scoped(toolkit: Toolkit) -> None:
    message = run(toolkit, LIST_CUSTOMER_ORDERS, ALICE, customer_id="CUST-1002")
    assert message.status == "error"
    unscoped = run(toolkit, LIST_CUSTOMER_ORDERS, customer_id="cust-1002", limit=1)
    assert len(payload(unscoped)["orders"]) == 1


def test_list_orders_limit_is_bounded(toolkit: Toolkit) -> None:
    assert run(toolkit, LIST_CUSTOMER_ORDERS, ALICE, limit=50).status == "error"


# -- refunds ---------------------------------------------------------------------------


def test_check_refund_eligibility(toolkit: Toolkit) -> None:
    data = payload(run(toolkit, CHECK_REFUND_ELIGIBILITY, ALICE, order_id="ORD-10002"))
    assert data["eligible"] is True
    assert data["refundable_amount"] == "349.00"
    assert data["requires_human_approval"] is True

    partial = payload(
        run(toolkit, CHECK_REFUND_ELIGIBILITY, ALICE, order_id="ORD-10002", amount="50")
    )
    assert partial["requires_human_approval"] is False
    assert partial["requested_amount"] == "50.00"


@pytest.mark.parametrize("amount", [0, -5, "12.345", "abc"])
def test_refund_amount_validation(toolkit: Toolkit, amount: object) -> None:
    message = run(toolkit, ISSUE_REFUND, order_id="ORD-10001", amount=amount, reason="Broken")
    assert message.status == "error"
    assert "Invalid arguments for issue_refund: amount" in payload(message)["error"]


def test_issue_refund_auto_approves_small_amounts(
    toolkit: Toolkit, repository: SupportRepository
) -> None:
    data = payload(
        run(toolkit, ISSUE_REFUND, ALICE, order_id="ORD-10001", amount=89.99, reason="Broken")
    )
    assert data["approved_by"] == "auto"
    assert repository.get_order("ORD-10001").refunded_cents == 8999


def test_issue_refund_above_threshold_needs_reviewer(toolkit: Toolkit) -> None:
    blocked = run(toolkit, ISSUE_REFUND, ALICE, order_id="ORD-10002", amount=349, reason="Wobbly")
    assert blocked.status == "error"
    assert "human reviewer" in payload(blocked)["error"]

    reviewed = ToolContext(customer_id="CUST-1001", reviewer="sam")
    data = payload(
        run(toolkit, ISSUE_REFUND, reviewed, order_id="ORD-10002", amount=349, reason="Wobbly")
    )
    assert data["approved_by"] == "sam"
    assert data["refund_id"].startswith("REF-")


def test_issue_refund_reports_rule_violations(toolkit: Toolkit) -> None:
    message = run(toolkit, ISSUE_REFUND, order_id="ORD-10005", amount=10, reason="Late")
    assert message.status == "error"
    assert "refund window ended" in payload(message)["error"]


# -- knowledge base and tickets ----------------------------------------------------------


def test_search_knowledge_base(toolkit: Toolkit) -> None:
    data = payload(run(toolkit, SEARCH_KNOWLEDGE_BASE, query="reset my password", k=1))
    assert [r["article"] for r in data["results"]] == ["password-reset"]
    assert run(toolkit, SEARCH_KNOWLEDGE_BASE, query="x").status == "error"


def test_create_ticket_uses_conversation_customer(
    toolkit: Toolkit, repository: SupportRepository
) -> None:
    data = payload(
        run(
            toolkit,
            CREATE_SUPPORT_TICKET,
            ALICE,
            subject="Parcel missing",
            description="Tracking says delivered but nothing arrived.",
            category="order_status",
            priority="high",
            order_id="ORD-10003",
        )
    )
    assert data["ticket_id"] == "TCK-000001"
    ticket = repository.get_ticket("TCK-000001")
    assert ticket.customer_id == "CUST-1001"
    assert ticket.priority == "high"
    assert ticket.status == "open"


def test_create_ticket_infers_customer_from_order(toolkit: Toolkit) -> None:
    data = payload(
        run(
            toolkit,
            CREATE_SUPPORT_TICKET,
            subject="Question",
            description="A question about my monitor order.",
            category="other",
            order_id="ORD-10004",
        )
    )
    assert data["customer_id"] == "CUST-1002"


def test_create_ticket_validation(toolkit: Toolkit) -> None:
    message = run(
        toolkit, CREATE_SUPPORT_TICKET, subject="Hi", description="short", category="sales"
    )
    assert message.status == "error"
    error = payload(message)["error"]
    assert "subject" in error and "description" in error and "category" in error


def test_get_ticket_rejects_malformed_ids(repository: SupportRepository) -> None:
    with pytest.raises(NotFoundError):
        repository.get_ticket("REF-000001")
    with pytest.raises(NotFoundError):
        repository.get_ticket("TCK-999999")
