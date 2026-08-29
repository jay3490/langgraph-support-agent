"""End-to-end turns through the real graph with the credential-free demo policy."""

from __future__ import annotations

from typing import Any

import pytest

from support_agent.approvals import ReviewDecision
from support_agent.config import Settings
from support_agent.demo_policy import DemoPolicy, classify
from support_agent.llm import FakeChatModel
from support_agent.runtime import SupportRuntime, collect, open_runtime
from tests.conftest import TODAY

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I want a refund for ORD-10001", "billing"),
        ("Why was I charged twice?", "billing"),
        ("Where is my order ORD-10003?", "order_status"),
        ("Can I cancel ORD-10006", "order_status"),
        ("I forgot my password", "technical"),
        ("The app crashes when I open it", "technical"),
        ("I want to speak to a manager right now", "escalate"),
        ("ORD-10001", "order_status"),
        ("hello there", "technical"),
        # A tie between order status and billing goes to order status first.
        ("Where is ORD-10001? I want a refund", "order_status"),
    ],
)
def test_classify(text: str, category: str) -> None:
    assert classify(text)[0] == category


def test_keywords_match_whole_words() -> None:
    assert classify("I am happy with the apple")[1] == []


@pytest.fixture
async def runtime(settings: Settings):
    async with open_runtime(
        settings, model=FakeChatModel(responder=DemoPolicy()), today=TODAY
    ) as rt:
        yield rt


async def turn(rt: SupportRuntime, text: str, customer: str | None = None) -> dict[str, Any]:
    thread = await rt.create_thread(customer)
    events = await collect(await rt.send_message(thread, text))
    done = events[-1].data
    done["nodes"] = [e.data["node"] for e in events if e.event == "node"]
    return done


async def test_order_status(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "Where is my order ORD-10003?", "CUST-1001")
    assert done["route"] == "order_status"
    assert "on its way with UPS" in done["reply"]
    assert "1Z999AA10000000003" in done["reply"]


async def test_order_of_another_customer_is_not_found(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "Where is ORD-10004?", "CUST-1001")
    assert "couldn't find that order" in done["reply"]


async def test_cancel_hint_for_processing_order(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "Can I cancel ORD-10006?", "CUST-1003")
    assert "can still be cancelled" in done["reply"]


async def test_small_refund_is_issued(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "I want a refund for ORD-10001", "CUST-1001")
    assert done["status"] == "completed"
    assert "refunded 89.99 USD" in done["reply"]
    assert runtime.repository.get_order("ORD-10001").refunded_cents == 8999


async def test_partial_refund_amount_from_message(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "Please refund $20 on ORD-10009", "CUST-1002")
    assert "refunded 20.00 USD" in done["reply"]


async def test_ineligible_refund_explains_why(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "refund ORD-10005 please", "CUST-1003")
    assert "can't be refunded" in done["reply"]
    assert "refund window ended" in done["reply"]


async def test_large_refund_goes_through_approval(runtime: SupportRuntime) -> None:
    thread = await runtime.create_thread("CUST-1001")
    events = await collect(await runtime.send_message(thread, "Refund ORD-10002, it wobbles"))
    assert events[-1].data["status"] == "awaiting_approval"
    assert events[-1].data["pending_approval"]["refunds"][0]["amount"] == "349.00"

    decision = ReviewDecision(approved=True, reviewer="lee")
    done = (await collect(await runtime.resume(thread, decision)))[-1].data
    assert "refunded 349.00 USD" in done["reply"]


async def test_rejected_refund(runtime: SupportRuntime) -> None:
    thread = await runtime.create_thread("CUST-1001")
    await collect(await runtime.send_message(thread, "Refund ORD-10002"))
    decision = ReviewDecision(approved=False, reviewer="lee", note="Needs photos first")
    done = (await collect(await runtime.resume(thread, decision)))[-1].data
    assert "wasn't able to issue the refund" in done["reply"]
    assert "Needs photos first" in done["reply"]


async def test_handoff_from_order_status_to_billing(runtime: SupportRuntime) -> None:
    done = await turn(
        runtime, "Where is ORD-10001? It was delivered damaged and I want a refund", "CUST-1001"
    )
    assert done["route"] == "order_status"
    assert done["agent"] == "billing"
    assert "refunded 89.99 USD" in done["reply"]
    assert done["nodes"].count("tools") == 3  # lookup, eligibility, refund


async def test_refund_without_order_number_lists_orders(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "I want a refund", "CUST-1001")
    assert "Which order should be refunded?" in done["reply"]
    assert "ORD-10002 (Standing desk, delivered)" in done["reply"]


async def test_refund_without_customer_asks_for_order(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "I want a refund")
    assert "ORD-12345" in done["reply"]


async def test_technical_answer_from_knowledge_base(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "I forgot my password and the reset link expired")
    assert done["route"] == "technical"
    assert '"Resetting your password"' in done["reply"]


async def test_technical_without_answer_opens_ticket(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "I found a crash bug", "CUST-1002")
    assert "opened ticket TCK-000001" in done["reply"]


async def test_escalation_opens_high_priority_ticket(runtime: SupportRuntime) -> None:
    done = await turn(runtime, "This is unacceptable, I want to speak to a manager", "CUST-1004")
    assert done["agent"] == "escalation"
    assert "TCK-000001" in done["reply"]
    ticket = runtime.repository.get_ticket("TCK-000001")
    assert ticket.priority == "high"
    assert ticket.customer_id == "CUST-1004"
