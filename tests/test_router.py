from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from support_agent.llm import FakeChatModel, ScriptedResponder, ai_tool_calls, tool_call
from support_agent.router import FALLBACK_REASON, make_router, router_context
from support_agent.state import SupportState

pytestmark = pytest.mark.anyio


def route_reply(category: str, reason: str = "because") -> AIMessage:
    return ai_tool_calls(tool_call("RouteDecision", {"category": category, "reason": reason}))


def state(*messages: object) -> SupportState:
    return SupportState(messages=list(messages))  # type: ignore[typeddict-item]


@pytest.mark.parametrize(
    ("category", "specialist"),
    [
        ("billing", "billing"),
        ("technical", "technical"),
        ("order_status", "order_status"),
        ("escalate", "escalation"),
    ],
)
async def test_router_sends_each_category_to_its_specialist(category: str, specialist: str) -> None:
    router = make_router(FakeChatModel(responder=ScriptedResponder([route_reply(category)])))
    command = await router(state(HumanMessage("help")))
    assert command.goto == specialist
    assert command.update["route"] == category
    assert command.update["active_agent"] == specialist
    assert command.update["visited"] == [specialist]


async def test_router_resets_turn_counters() -> None:
    router = make_router(FakeChatModel(responder=ScriptedResponder([route_reply("billing")])))
    command = await router(state(HumanMessage("refund please")))
    update = command.update
    assert update["tool_calls_this_turn"] == 0
    assert update["handoffs"] == 0
    assert update["step_limit_hit"] is False
    assert update["refund_decisions"] == {}
    assert update["final_response"] is None


async def test_router_binds_only_the_route_schema() -> None:
    responder = ScriptedResponder([route_reply("technical")])
    router = make_router(FakeChatModel(responder=responder))
    await router(state(HumanMessage("my app crashes")))
    prompt, tools = responder.calls[0]
    assert tools == ["RouteDecision"]
    assert isinstance(prompt[0], SystemMessage)
    assert prompt[-1].content == "my app crashes"


@pytest.mark.parametrize(
    "reply",
    [
        route_reply("sales"),  # not a valid category
        AIMessage(content="I think this is billing"),  # no tool call at all
    ],
)
async def test_unparseable_output_falls_back_to_escalation(reply: AIMessage) -> None:
    router = make_router(FakeChatModel(responder=ScriptedResponder([reply])))
    command = await router(state(HumanMessage("???")))
    assert command.goto == "escalation"
    assert command.update["route_reason"] == FALLBACK_REASON


def test_router_context_drops_tool_traffic_and_keeps_recent_turns() -> None:
    call = tool_call("lookup_order", {"order_id": "ORD-10001"}, id="c1")
    messages = [
        HumanMessage("one"),
        ai_tool_calls(call),
        ToolMessage(content="{}", tool_call_id="c1"),
        AIMessage(content="answer one"),
        *[HumanMessage(f"msg {i}") for i in range(6)],
    ]
    context = router_context(messages)
    assert len(context) == 6
    assert context[-1].content == "msg 5"
    assert all(not isinstance(m, ToolMessage) for m in router_context(messages, limit=20))
    assert [m.content for m in router_context(messages, limit=20)][:2] == ["one", "answer one"]
