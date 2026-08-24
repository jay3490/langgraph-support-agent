"""Builders for scripted model replies."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from support_agent.llm import ai_tool_calls, tool_call


def route(category: str, reason: str = "test routing") -> AIMessage:
    return ai_tool_calls(tool_call("RouteDecision", {"category": category, "reason": reason}))


def calls(name: str, *, id: str | None = None, **args: Any) -> AIMessage:
    return ai_tool_calls(tool_call(name, args, id=id))


def transfer(target: str, reason: str = "better placed") -> AIMessage:
    return calls("transfer_to_specialist", target=target, reason=reason)


def tool_results(messages: list[BaseMessage]) -> list[tuple[str | None, dict[str, Any]]]:
    return [(m.name, json.loads(str(m.content))) for m in messages if isinstance(m, ToolMessage)]
