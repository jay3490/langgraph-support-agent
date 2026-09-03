"""Run support threads: create them, stream turns, inspect them and resume approvals.

The runtime owns the compiled graph and its SQLite checkpointer. Both the HTTP API and
the terminal chat use it, so they behave identically.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from support_agent.approvals import ReviewDecision
from support_agent.config import Settings
from support_agent.db import create_demo_database
from support_agent.graph import SupportWorkflow
from support_agent.kb import KnowledgeBase
from support_agent.llm import build_chat_model
from support_agent.money import to_cents
from support_agent.repository import RefundPolicy, SupportRepository
from support_agent.state import SPECIALISTS
from support_agent.tools import Toolkit

logger = logging.getLogger(__name__)

EventKind = Literal["node", "token", "approval_required", "done"]


class ThreadNotFoundError(LookupError):
    pass


class ThreadBusyError(RuntimeError):
    """Another turn is already running on this thread."""


class ApprovalPendingError(RuntimeError):
    """The thread is paused on a refund approval; approve or reject it first."""


class NoPendingApprovalError(RuntimeError):
    pass


@dataclass(frozen=True)
class TurnEvent:
    """One event of a running turn.

    * ``node``: a graph node finished; ``data`` summarises its state update.
    * ``token``: a piece of a specialist's reply as the model streams it.
    * ``approval_required``: the turn paused for a human reviewer.
    * ``done``: the turn ended (``status`` is ``completed`` or ``awaiting_approval``).
    """

    event: EventKind
    data: dict[str, Any]


@dataclass(frozen=True)
class ThreadView:
    thread_id: str
    customer_id: str | None
    created_at: str | None
    route: str | None
    active_agent: str | None
    final_response: str | None
    pending_approval: dict[str, Any] | None
    messages: list[dict[str, Any]] = field(default_factory=list)


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"type": message.type, "content": message.text}
    if message.name:
        data["name"] = message.name
    if isinstance(message, AIMessage) and message.tool_calls:
        data["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        data["tool_call_id"] = message.tool_call_id
        data["status"] = message.status
    return data


def summarize_update(node: str, update: dict[str, Any] | None) -> dict[str, Any]:
    """A compact, JSON-friendly description of what a node changed."""
    update = update or {}
    summary: dict[str, Any] = {"node": node}
    if node == "router":
        summary.update(
            route=update.get("route"),
            agent=update.get("active_agent"),
            reason=update.get("route_reason"),
        )
    messages = update.get("messages") or []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            summary["tool_calls"] = [c["name"] for c in message.tool_calls]
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    if tool_results:
        summary["tool_results"] = [{"tool": m.name, "status": m.status} for m in tool_results]
    if node not in ("router",) and "active_agent" in update:
        summary["handoff_to"] = update["active_agent"]
    if update.get("step_limit_hit"):
        summary["step_limit_hit"] = True
    if "refund_decisions" in update and node == "approval":
        summary["decisions"] = list(update["refund_decisions"].values())
    return summary


class SupportRuntime:
    def __init__(
        self, graph: CompiledStateGraph, repository: SupportRepository, settings: Settings
    ) -> None:
        self.graph = graph
        self.repository = repository
        self.settings = settings
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _config(thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    async def _snapshot(self, thread_id: str) -> StateSnapshot:
        snapshot = await self.graph.aget_state(self._config(thread_id))
        if snapshot.created_at is None:
            raise ThreadNotFoundError(thread_id)
        return snapshot

    @staticmethod
    def _pending(snapshot: StateSnapshot) -> dict[str, Any] | None:
        for pending in snapshot.interrupts:
            if isinstance(pending.value, dict):
                return pending.value
        return None

    def _lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.setdefault(thread_id, asyncio.Lock())
        if lock.locked():
            raise ThreadBusyError(thread_id)
        return lock

    # -- public API -----------------------------------------------------------------------

    async def create_thread(self, customer_id: str | None = None) -> str:
        """Open a thread, optionally for a known customer; returns its id."""
        if customer_id is not None:
            await asyncio.to_thread(self.repository.get_customer, customer_id)
        thread_id = uuid.uuid4().hex
        await self.graph.aupdate_state(
            self._config(thread_id), {"messages": [], "customer_id": customer_id}, as_node=START
        )
        logger.info("Opened thread %s", thread_id)
        return thread_id

    async def get_thread(self, thread_id: str) -> ThreadView:
        snapshot = await self._snapshot(thread_id)
        values = snapshot.values
        return ThreadView(
            thread_id=thread_id,
            customer_id=values.get("customer_id"),
            created_at=snapshot.created_at,
            route=values.get("route"),
            active_agent=values.get("active_agent"),
            final_response=values.get("final_response"),
            pending_approval=self._pending(snapshot),
            messages=[serialize_message(m) for m in values.get("messages", [])],
        )

    async def send_message(self, thread_id: str, text: str) -> AsyncIterator[TurnEvent]:
        """Validate the thread, then return the event stream of a new turn."""
        snapshot = await self._snapshot(thread_id)
        if self._pending(snapshot) is not None:
            raise ApprovalPendingError(thread_id)
        lock = self._lock(thread_id)
        # Customer text can contain personal data; the logging filter masks it.
        logger.info("Thread %s received: %s", thread_id, text)
        return self._stream(thread_id, {"messages": [HumanMessage(text)]}, lock)

    async def resume(self, thread_id: str, decision: ReviewDecision) -> AsyncIterator[TurnEvent]:
        """Validate the pending approval, then return the event stream of the resumed turn."""
        snapshot = await self._snapshot(thread_id)
        if self._pending(snapshot) is None:
            raise NoPendingApprovalError(thread_id)
        lock = self._lock(thread_id)
        return self._stream(thread_id, Command(resume=decision.model_dump()), lock)

    async def _stream(
        self, thread_id: str, graph_input: Any, lock: asyncio.Lock
    ) -> AsyncIterator[TurnEvent]:
        config = self._config(thread_id)
        async with lock:
            stream = self.graph.astream(graph_input, config, stream_mode=["updates", "messages"])
            async for mode, chunk in stream:
                if mode == "messages":
                    message, metadata = chunk
                    node = metadata.get("langgraph_node")
                    if isinstance(message, AIMessageChunk) and node in SPECIALISTS and message.text:
                        yield TurnEvent("token", {"node": node, "content": message.text})
                    continue
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        for pending in update:
                            yield TurnEvent("approval_required", pending.value)
                    else:
                        yield TurnEvent("node", summarize_update(node, update))
            snapshot = await self.graph.aget_state(config)
        yield TurnEvent("done", self._result(thread_id, snapshot))

    def _result(self, thread_id: str, snapshot: StateSnapshot) -> dict[str, Any]:
        values = snapshot.values
        pending = self._pending(snapshot)
        return {
            "thread_id": thread_id,
            "status": "awaiting_approval" if pending is not None else "completed",
            "reply": None if pending is not None else values.get("final_response"),
            "route": values.get("route"),
            "agent": values.get("active_agent"),
            "pending_approval": pending,
        }


async def collect(events: AsyncIterator[TurnEvent]) -> list[TurnEvent]:
    return [event async for event in events]


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    today: date | None = None,
) -> AsyncIterator[SupportRuntime]:
    """Build everything the graph needs and keep the checkpointer open while in use.

    ``today`` pins the date used for seeding and for the refund rules (tests use it).
    """
    if settings.seed_on_startup:
        if create_demo_database(settings.database_path, today=today):
            logger.info("Seeded demo database at %s", settings.database_path)
    elif not settings.database_path.exists():
        raise FileNotFoundError(
            f"{settings.database_path} does not exist; run `support-agent seed-db` first."
        )

    clock: Callable[[], date] = (lambda: today) if today is not None else date.today
    policy = RefundPolicy(
        window_days=settings.refund_window_days,
        gold_window_days=settings.gold_refund_window_days,
        approval_threshold_cents=to_cents(settings.refund_approval_threshold),
    )
    repository = SupportRepository(settings.database_path, policy=policy, today=clock)
    workflow = SupportWorkflow(
        model or build_chat_model(settings),
        Toolkit(repository, KnowledgeBase.from_directory(settings.kb_dir)),
        max_tool_steps=settings.max_tool_steps,
        max_handoffs=settings.max_handoffs,
    )
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as checkpointer:
        yield SupportRuntime(workflow.compile(checkpointer=checkpointer), repository, settings)
