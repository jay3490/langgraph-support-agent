"""Human-in-the-loop refunds: interrupt, approve or reject, resume, and restarts."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from support_agent.approvals import ReviewDecision
from support_agent.config import Settings
from support_agent.llm import FakeChatModel, ScriptedResponder
from support_agent.repository import NotFoundError, SupportRepository
from support_agent.runtime import (
    ApprovalPendingError,
    NoPendingApprovalError,
    ThreadNotFoundError,
    TurnEvent,
    collect,
    open_runtime,
)
from tests.conftest import TODAY
from tests.helpers import calls, route

pytestmark = pytest.mark.anyio

BIG_REFUND = calls(
    "issue_refund", id="refund-1", order_id="ORD-10002", amount=349, reason="Desk wobbles"
)
APPROVE = ReviewDecision(approved=True, reviewer="sam", note="Photos confirm the damage")
REJECT = ReviewDecision(approved=False, reviewer="sam", note="Outside policy")


def scripted(*replies: AIMessage | str) -> FakeChatModel:
    return FakeChatModel(responder=ScriptedResponder(replies))


def by_kind(events: list[TurnEvent], kind: str) -> list[dict]:
    return [e.data for e in events if e.event == kind]


async def test_small_refund_needs_no_approval(settings: Settings) -> None:
    model = scripted(
        route("billing"),
        calls("issue_refund", order_id="ORD-10001", amount="89.99", reason="Arrived broken"),
        "Done, your refund is on its way.",
    )
    async with open_runtime(settings, model=model, today=TODAY) as runtime:
        thread = await runtime.create_thread("CUST-1001")
        events = await collect(await runtime.send_message(thread, "Refund ORD-10001 please"))
        assert by_kind(events, "approval_required") == []
        [done] = by_kind(events, "done")
        assert done["status"] == "completed"
        assert done["reply"] == "Done, your refund is on its way."
        assert runtime.repository.get_order("ORD-10001").refunded_cents == 8999


async def test_large_refund_pauses_until_approved(settings: Settings) -> None:
    model = scripted(route("billing"), BIG_REFUND, "Your refund of 349.00 was approved.")
    async with open_runtime(settings, model=model, today=TODAY) as runtime:
        thread = await runtime.create_thread("CUST-1001")
        events = await collect(await runtime.send_message(thread, "Refund my desk ORD-10002"))

        [request] = by_kind(events, "approval_required")
        assert request["type"] == "refund_approval"
        assert request["threshold"] == "100.00"
        assert request["refunds"] == [
            {
                "tool_call_id": "refund-1",
                "order_id": "ORD-10002",
                "amount": "349.00",
                "reason": "Desk wobbles",
            }
        ]
        [done] = by_kind(events, "done")
        assert done["status"] == "awaiting_approval"
        assert done["reply"] is None
        assert runtime.repository.get_order("ORD-10002").refunded_cents == 0

        view = await runtime.get_thread(thread)
        assert view.pending_approval == request
        with pytest.raises(ApprovalPendingError):
            await runtime.send_message(thread, "hello?")

        resumed = await collect(await runtime.resume(thread, APPROVE))
        [done] = by_kind(resumed, "done")
        assert done["status"] == "completed"
        assert done["reply"] == "Your refund of 349.00 was approved."
        [approval] = [n for n in by_kind(resumed, "node") if n["node"] == "approval"]
        assert approval["decisions"][0]["reviewer"] == "sam"

        order = runtime.repository.get_order("ORD-10002")
        assert order.refunded_cents == 34900
        assert (await runtime.get_thread(thread)).pending_approval is None


async def test_rejected_refund_is_reported_to_the_agent(settings: Settings) -> None:
    responder = ScriptedResponder(
        [route("billing"), BIG_REFUND, "Sorry, the refund was not approved."]
    )
    async with open_runtime(settings, model=FakeChatModel(responder=responder), today=TODAY) as rt:
        thread = await rt.create_thread("CUST-1001")
        await collect(await rt.send_message(thread, "Refund ORD-10002"))
        events = await collect(await rt.resume(thread, REJECT))
        assert by_kind(events, "done")[0]["reply"] == "Sorry, the refund was not approved."
        assert rt.repository.get_order("ORD-10002").refunded_cents == 0
        # The billing agent saw the rejection as the tool result.
        last_prompt = responder.calls[-1][0]
        assert "rejected by reviewer sam: Outside policy" in str(last_prompt[-1].content)


async def test_paused_thread_survives_a_restart(settings: Settings) -> None:
    async with open_runtime(
        settings, model=scripted(route("billing"), BIG_REFUND), today=TODAY
    ) as runtime:
        thread = await runtime.create_thread("CUST-1001")
        events = await collect(await runtime.send_message(thread, "Refund ORD-10002"))
        assert by_kind(events, "done")[0]["status"] == "awaiting_approval"

    # A new process: fresh model, fresh checkpointer connection, same files on disk.
    async with open_runtime(settings, model=scripted("Approved and refunded."), today=TODAY) as rt:
        view = await rt.get_thread(thread)
        assert view.pending_approval is not None
        assert view.customer_id == "CUST-1001"
        events = await collect(await rt.resume(thread, APPROVE))
        assert by_kind(events, "done")[0]["reply"] == "Approved and refunded."
        assert rt.repository.get_order("ORD-10002").refunded_cents == 34900


async def test_refund_is_issued_once_even_though_the_node_reruns(settings: Settings) -> None:
    async with open_runtime(
        settings, model=scripted(route("billing"), BIG_REFUND, "ok"), today=TODAY
    ) as runtime:
        thread = await runtime.create_thread("CUST-1001")
        await collect(await runtime.send_message(thread, "Refund ORD-10002"))
        await collect(await runtime.resume(thread, APPROVE))
        repo = SupportRepository(settings.database_path, today=lambda: TODAY)
        assert repo.get_order("ORD-10002").refunded_cents == 34900


async def test_thread_errors(settings: Settings) -> None:
    async with open_runtime(settings, model=scripted(), today=TODAY) as runtime:
        with pytest.raises(ThreadNotFoundError):
            await runtime.get_thread("missing")
        with pytest.raises(ThreadNotFoundError):
            await runtime.send_message("missing", "hi")
        with pytest.raises(NotFoundError):
            await runtime.create_thread("CUST-9999")
        thread = await runtime.create_thread()
        with pytest.raises(NoPendingApprovalError):
            await runtime.resume(thread, APPROVE)


async def test_tokens_and_node_updates_are_streamed(settings: Settings) -> None:
    model = scripted(route("technical"), "Open Settings and choose Security.")
    async with open_runtime(settings, model=model, today=TODAY) as runtime:
        thread = await runtime.create_thread()
        events = await collect(await runtime.send_message(thread, "How do I turn on 2FA?"))
        tokens = by_kind(events, "token")
        assert "".join(t["content"] for t in tokens) == "Open Settings and choose Security."
        assert {t["node"] for t in tokens} == {"technical"}
        nodes = [n["node"] for n in by_kind(events, "node")]
        assert nodes == ["router", "technical", "supervisor", "respond"]
        assert by_kind(events, "node")[0]["route"] == "technical"
        assert events[-1].event == "done"


async def test_messages_accumulate_across_turns(settings: Settings) -> None:
    model = scripted(route("technical"), "First answer.", route("technical"), "Second answer.")
    async with open_runtime(settings, model=model, today=TODAY) as runtime:
        thread = await runtime.create_thread()
        await collect(await runtime.send_message(thread, "one"))
        await collect(await runtime.send_message(thread, "two"))
        view = await runtime.get_thread(thread)
        assert [m["content"] for m in view.messages] == [
            "one",
            "First answer.",
            "two",
            "Second answer.",
        ]
        assert view.final_response == "Second answer."
