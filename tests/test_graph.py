from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from support_agent.graph import AGENT_TOOLS, TRANSFER_TOOL, SupportWorkflow
from support_agent.llm import FakeChatModel, ScriptedResponder, bound_tool_names
from support_agent.repository import SupportRepository
from support_agent.tools import Toolkit
from tests.helpers import calls, route, tool_results, transfer

pytestmark = pytest.mark.anyio


def workflow(
    toolkit: Toolkit, replies: list[AIMessage | str], **kwargs: Any
) -> tuple[SupportWorkflow, ScriptedResponder]:
    responder = ScriptedResponder(replies)
    return SupportWorkflow(FakeChatModel(responder=responder), toolkit, **kwargs), responder


async def run(flow: SupportWorkflow, text: str, customer_id: str | None = None) -> dict[str, Any]:
    graph = flow.compile()
    return await graph.ainvoke({"messages": [HumanMessage(text)], "customer_id": customer_id})


async def test_order_status_turn(toolkit: Toolkit) -> None:
    flow, responder = workflow(
        toolkit,
        [
            route("order_status"),
            calls("lookup_order", order_id="ORD-10003"),
            "Your cable is on its way with UPS.",
        ],
    )
    state = await run(flow, "Where is ORD-10003?", "CUST-1001")
    assert state["final_response"] == "Your cable is on its way with UPS."
    assert state["route"] == "order_status"
    assert state["tool_calls_this_turn"] == 1
    [(name, result)] = tool_results(state["messages"])
    assert name == "lookup_order"
    assert result["status"] == "shipped"
    assert responder.remaining == 0
    # The specialist saw only its own tools plus the handoff tool.
    assert set(responder.calls[1][1]) == {*AGENT_TOOLS["order_status"], TRANSFER_TOOL}
    assert state["messages"][-1].name == "order_status"


async def test_specialist_prompt_names_the_customer(toolkit: Toolkit) -> None:
    flow, responder = workflow(toolkit, [route("technical"), "Try resetting it."])
    await run(flow, "password?", "CUST-1002")
    system = responder.calls[1][0][0]
    assert "Role: technical" in str(system.content)
    assert "CUST-1002" in str(system.content)


async def test_tool_errors_go_back_to_the_model(toolkit: Toolkit) -> None:
    flow, _ = workflow(
        toolkit,
        [
            route("order_status"),
            calls("lookup_order", order_id="10003"),
            calls("lookup_order", order_id="ORD-10003"),
            "Found it.",
        ],
    )
    state = await run(flow, "Where is 10003?")
    first, second = tool_results(state["messages"])
    assert "Invalid arguments" in first[1]["error"]
    assert second[1]["order_id"] == "ORD-10003"
    assert state["final_response"] == "Found it."


async def test_handoff_between_specialists(toolkit: Toolkit) -> None:
    flow, responder = workflow(
        toolkit,
        [
            route("order_status"),
            transfer("billing", "customer wants a refund"),
            "Billing here, I can help with that refund.",
        ],
    )
    state = await run(flow, "Where is my order and can I get a refund?")
    assert state["active_agent"] == "billing"
    assert state["visited"] == ["order_status", "billing"]
    assert state["handoffs"] == 1
    assert tool_results(state["messages"])[0][1]["transferred_to"] == "billing"
    assert "issue_refund" in responder.calls[2][1]  # billing's tools are bound now
    assert state["final_response"].startswith("Billing here")


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        ([route("billing"), transfer("billing"), "ok"], "already"),
        (
            [route("billing"), transfer("technical"), transfer("billing"), "ok"],
            "already handled this turn",
        ),
        ([route("billing"), transfer("nobody"), "ok"], "Invalid transfer"),
    ],
)
async def test_refused_handoffs(
    toolkit: Toolkit, replies: list[AIMessage | str], expected: str
) -> None:
    flow, _ = workflow(toolkit, replies)
    state = await run(flow, "help")
    errors = [r["error"] for _, r in tool_results(state["messages"]) if "error" in r]
    assert expected in errors[-1]
    assert state["final_response"] == "ok"


async def test_handoff_limit(toolkit: Toolkit) -> None:
    flow, _ = workflow(
        toolkit,
        [route("billing"), transfer("technical"), transfer("order_status"), "ok"],
        max_handoffs=1,
    )
    state = await run(flow, "help")
    assert state["active_agent"] == "technical"
    assert "No more handoffs" in tool_results(state["messages"])[-1][1]["error"]


async def test_other_calls_next_to_a_transfer_are_skipped(toolkit: Toolkit) -> None:
    mixed = AIMessage(
        content="",
        tool_calls=[
            {"name": TRANSFER_TOOL, "args": {"target": "billing", "reason": "refund"}, "id": "t"},
            {"name": "lookup_order", "args": {"order_id": "ORD-10001"}, "id": "l"},
        ],
    )
    flow, _ = workflow(toolkit, [route("order_status"), mixed, "ok"])
    state = await run(flow, "help")
    [transferred, skipped] = tool_results(state["messages"])
    assert transferred[1]["transferred_to"] == "billing"
    assert "transferred to billing" in skipped[1]["error"]


def _always_searching(
    messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
) -> AIMessage:
    if "RouteDecision" in bound_tool_names(tools):
        return route("technical")
    return calls("search_knowledge_base", query="app sync")


async def test_step_limit_ends_the_turn_with_a_ticket(
    toolkit: Toolkit, repository: SupportRepository
) -> None:
    flow = SupportWorkflow(FakeChatModel(responder=_always_searching), toolkit, max_tool_steps=3)
    state = await run(flow, "my app will not sync", "CUST-1003")
    results = tool_results(state["messages"])
    executed = [r for _, r in results if "results" in r]
    assert len(executed) == 3
    assert "limit of 3 tool calls" in results[-1][1]["error"]
    assert state["step_limit_hit"] is True
    assert "TCK-000001" in state["final_response"]
    ticket = repository.get_ticket("TCK-000001")
    assert ticket.priority == "high"
    assert ticket.customer_id == "CUST-1003"
    assert "my app will not sync" in ticket.description


async def test_empty_reply_gets_a_fallback(toolkit: Toolkit) -> None:
    flow, _ = workflow(toolkit, [route("technical"), ""])
    state = await run(flow, "hmm")
    assert state["final_response"].startswith("Sorry")


def test_graph_shape(toolkit: Toolkit) -> None:
    flow, _ = workflow(toolkit, [])
    mermaid = flow.compile().get_graph().draw_mermaid()
    for node in ("router", "billing", "technical", "order_status", "escalation", "supervisor"):
        assert node in mermaid
    assert "supervisor -.-> tools" in mermaid
    assert "respond --> __end__" in mermaid
