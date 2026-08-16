from __future__ import annotations

from typing import Literal

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from support_agent.config import Settings
from support_agent.llm import (
    FakeChatModel,
    ScriptedResponder,
    ScriptExhaustedError,
    ai_tool_calls,
    build_chat_model,
    tool_call,
)


class Pick(BaseModel):
    """Pick a colour."""

    colour: Literal["red", "blue"]


def test_scripted_text_replies_in_order() -> None:
    responder = ScriptedResponder(["first", "second"])
    model = FakeChatModel(responder=responder)
    assert model.invoke("hi").content == "first"
    assert model.invoke([HumanMessage("again")]).content == "second"
    assert responder.remaining == 0
    with pytest.raises(ScriptExhaustedError):
        model.invoke("more")


def test_tool_calls_get_ids_and_tools_are_visible() -> None:
    responder = ScriptedResponder([ai_tool_calls(tool_call("lookup", {"q": 1}))])
    model = FakeChatModel(responder=responder).bind_tools([Pick])
    reply = model.invoke("hi")
    assert isinstance(reply, AIMessage)
    assert reply.tool_calls[0]["name"] == "lookup"
    assert reply.tool_calls[0]["id"]
    assert responder.calls[0][1] == ["Pick"]


def test_structured_output_through_tool_calling() -> None:
    responder = ScriptedResponder([ai_tool_calls(tool_call("Pick", {"colour": "blue"}))])
    structured = FakeChatModel(responder=responder).with_structured_output(
        Pick, method="function_calling"
    )
    assert structured.invoke("which?") == Pick(colour="blue")


def test_structured_output_parse_error_is_reported_with_include_raw() -> None:
    responder = ScriptedResponder([ai_tool_calls(tool_call("Pick", {"colour": "green"}))])
    structured = FakeChatModel(responder=responder).with_structured_output(Pick, include_raw=True)
    result = structured.invoke("which?")
    assert result["parsed"] is None
    assert result["parsing_error"] is not None


def test_streaming_text_token_by_token() -> None:
    model = FakeChatModel(responder=ScriptedResponder(["Your order has shipped."]))
    chunks = list(model.stream("status?"))
    assert [c.content for c in chunks if c.content] == ["Your ", "order ", "has ", "shipped."]
    assert all(isinstance(c, AIMessageChunk) for c in chunks)


def test_streaming_tool_calls_reassemble() -> None:
    reply = ai_tool_calls(tool_call("lookup", {"order_id": "ORD-10001"}, id="c1"))
    model = FakeChatModel(responder=ScriptedResponder([reply]))
    merged: AIMessageChunk | None = None
    for chunk in model.stream("status?"):
        merged = chunk if merged is None else merged + chunk
    assert merged is not None
    assert merged.tool_calls == [
        {"name": "lookup", "args": {"order_id": "ORD-10001"}, "id": "c1", "type": "tool_call"}
    ]


def test_factory_builds_fake_with_responder() -> None:
    model = build_chat_model(Settings(_env_file=None), responder=ScriptedResponder(["ok"]))
    assert isinstance(model, FakeChatModel)
    with pytest.raises(ValueError, match="responder"):
        build_chat_model(Settings(_env_file=None))


def test_factory_builds_openai_compatible_client() -> None:
    settings = Settings(
        _env_file=None,
        provider="openai",
        model="llama3.2",
        openai_base_url="http://localhost:11434/v1",
    )
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "llama3.2"
    assert model.openai_api_base == "http://localhost:11434/v1"


def test_factory_requires_key_for_hosted_openai() -> None:
    with pytest.raises(ValueError, match="SUPPORT_OPENAI_API_KEY"):
        build_chat_model(Settings(_env_file=None, provider="openai"))
