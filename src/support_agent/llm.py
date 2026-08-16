"""Chat model providers.

``fake`` is a deterministic chat model whose replies come from a *responder*: a
callable that receives the prompt messages and the tools bound to the call and returns
an :class:`AIMessage`, which may contain tool calls. It supports ``bind_tools``,
``with_structured_output`` (through the default tool-calling implementation) and token
streaming, so the graph runs unchanged with no network or credentials. ``openai`` talks
to any OpenAI-compatible endpoint, including a local Ollama server.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from support_agent.config import Settings

Responder = Callable[[Sequence[BaseMessage], Sequence[dict[str, Any]]], AIMessage]
"""Produce the next reply from the prompt messages and the bound tools (OpenAI format)."""

_STREAM_PIECE = re.compile(r"\S+\s*|\s+")


def tool_call(name: str, args: dict[str, Any] | None = None, *, id: str | None = None) -> ToolCall:
    return ToolCall(name=name, args=args or {}, id=id or f"call_{uuid.uuid4().hex[:12]}")


def ai_tool_calls(*calls: ToolCall, content: str = "") -> AIMessage:
    return AIMessage(content=content, tool_calls=list(calls))


def bound_tool_names(tools: Sequence[dict[str, Any]]) -> set[str]:
    return {tool["function"]["name"] for tool in tools}


class ScriptExhaustedError(RuntimeError):
    pass


class ScriptedResponder:
    """Reply with a fixed sequence of messages and record every prompt it receives."""

    def __init__(self, replies: Iterable[AIMessage | str]) -> None:
        self._replies: deque[AIMessage] = deque(
            AIMessage(content=r) if isinstance(r, str) else r for r in replies
        )
        self.calls: list[tuple[list[BaseMessage], list[str]]] = []

    @property
    def remaining(self) -> int:
        return len(self._replies)

    def __call__(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
    ) -> AIMessage:
        self.calls.append((list(messages), sorted(bound_tool_names(tools))))
        if not self._replies:
            raise ScriptExhaustedError("The scripted model has no replies left.")
        return self._replies.popleft()


class FakeChatModel(BaseChatModel):
    """A chat model driven by a :data:`Responder`; see the module docstring."""

    responder: Responder

    @property
    def _llm_type(self) -> str:
        return "fake-support"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _reply(
        self, messages: list[BaseMessage], tools: Sequence[dict[str, Any]] | None
    ) -> AIMessage:
        reply = self.responder(messages, tools or [])
        calls = [
            {**call, "id": call.get("id") or tool_call(call["name"])["id"]}
            for call in reply.tool_calls
        ]
        return reply.model_copy(
            update={"tool_calls": calls, "id": reply.id or f"run-{uuid.uuid4()}"}
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self._reply(messages, kwargs.get("tools"))
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._reply(messages, kwargs.get("tools"))
        text = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
        pieces = _STREAM_PIECE.findall(text)
        for piece in pieces:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece, id=reply.id))
            if run_manager is not None:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
        if reply.tool_calls or not pieces:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id=reply.id,
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                            "type": "tool_call_chunk",
                        }
                        for index, call in enumerate(reply.tool_calls)
                    ],
                    chunk_position="last",
                )
            )


def build_chat_model(settings: Settings, *, responder: Responder | None = None) -> BaseChatModel:
    """Create the chat model selected by ``settings.provider``.

    The ``fake`` provider needs a ``responder`` that scripts its replies.
    """
    if settings.provider == "fake":
        if responder is None:
            raise ValueError("The fake provider needs a responder to script its replies.")
        return FakeChatModel(responder=responder)

    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if api_key is None and settings.openai_base_url is None:
        raise ValueError(
            "SUPPORT_OPENAI_API_KEY is required for the openai provider unless "
            "SUPPORT_OPENAI_BASE_URL points at a server that needs no key (e.g. Ollama)."
        )
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.openai_base_url,
        # Local OpenAI-compatible servers such as Ollama ignore the key but the client needs one.
        api_key=api_key or "not-needed",
        temperature=settings.temperature,
    )
