"""FastAPI application exposing support threads over HTTP.

``POST /threads/{id}/messages`` streams the turn as Server-Sent Events: ``node`` events
as graph nodes finish, ``token`` events as a specialist writes its reply, an
``approval_required`` event when a refund needs review, and a final ``done`` event.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from support_agent import __version__
from support_agent.approvals import ReviewDecision
from support_agent.config import Settings, get_settings
from support_agent.repository import NotFoundError
from support_agent.runtime import (
    ApprovalPendingError,
    NoPendingApprovalError,
    SupportRuntime,
    ThreadBusyError,
    ThreadNotFoundError,
    TurnEvent,
    open_runtime,
)


class CreateThreadRequest(BaseModel):
    customer_id: str | None = Field(
        default=None,
        pattern=r"^CUST-\d{4}$",
        description="Customer the conversation is about; scopes the order tools to them.",
    )


class ThreadCreated(BaseModel):
    thread_id: str
    customer_id: str | None


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=500)


class TurnResult(BaseModel):
    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    reply: str | None
    route: str | None
    agent: str | None
    pending_approval: dict[str, Any] | None


class ThreadState(BaseModel):
    thread_id: str
    customer_id: str | None
    created_at: str | None
    route: str | None
    active_agent: str | None
    final_response: str | None
    pending_approval: dict[str, Any] | None
    messages: list[dict[str, Any]]


def _runtime(request: Request) -> SupportRuntime:
    return request.app.state.runtime


Runtime = Annotated[SupportRuntime, Depends(_runtime)]


async def _message_events(
    thread_id: str, body: MessageRequest, runtime: Runtime
) -> AsyncIterator[TurnEvent]:
    # Resolved as a dependency so that unknown or paused threads are rejected with a
    # proper status code before the event stream (and its 200 response) starts.
    return await runtime.send_message(thread_id, body.content)


async def _finish(events: AsyncIterator[TurnEvent]) -> TurnResult:
    done: dict[str, Any] = {}
    async for event in events:
        if event.event == "done":
            done = event.data
    return TurnResult.model_validate(done)


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


def create_app(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    today: date | None = None,
) -> FastAPI:
    """Create the app; ``model`` and ``today`` let tests inject a scripted model and clock."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_runtime(settings, model=model, today=today) as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(
        title="LangGraph support agent",
        version=__version__,
        description="Multi-agent customer support with human approval of large refunds.",
        lifespan=lifespan,
    )

    @app.exception_handler(ThreadNotFoundError)
    async def thread_not_found(_: Request, exc: ThreadNotFoundError) -> JSONResponse:
        return _error(404, f"Thread {exc} was not found.")

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, str(exc))

    @app.exception_handler(ApprovalPendingError)
    async def approval_pending(_: Request, exc: ApprovalPendingError) -> JSONResponse:
        return _error(409, "This thread is waiting for a refund approval; approve or reject it.")

    @app.exception_handler(NoPendingApprovalError)
    async def nothing_pending(_: Request, exc: NoPendingApprovalError) -> JSONResponse:
        return _error(409, "This thread has no pending approval.")

    @app.exception_handler(ThreadBusyError)
    async def busy(_: Request, exc: ThreadBusyError) -> JSONResponse:
        return _error(409, "A turn is already running on this thread.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "provider": settings.provider, "version": __version__}

    @app.post("/threads", status_code=201)
    async def create_thread(
        runtime: Runtime, body: CreateThreadRequest | None = None
    ) -> ThreadCreated:
        customer_id = body.customer_id if body is not None else None
        thread_id = await runtime.create_thread(customer_id)
        return ThreadCreated(thread_id=thread_id, customer_id=customer_id)

    @app.get("/threads/{thread_id}")
    async def get_thread(thread_id: str, runtime: Runtime) -> ThreadState:
        view = await runtime.get_thread(thread_id)
        return ThreadState.model_validate(view, from_attributes=True)

    @app.post("/threads/{thread_id}/messages", response_class=EventSourceResponse)
    async def post_message(
        events: Annotated[AsyncIterator[TurnEvent], Depends(_message_events)],
    ) -> AsyncIterable[ServerSentEvent]:
        async for event in events:
            yield ServerSentEvent(event=event.event, data=event.data)

    @app.post("/threads/{thread_id}/approve")
    async def approve(thread_id: str, body: ReviewRequest, runtime: Runtime) -> TurnResult:
        decision = ReviewDecision(approved=True, reviewer=body.reviewer, note=body.note)
        return await _finish(await runtime.resume(thread_id, decision))

    @app.post("/threads/{thread_id}/reject")
    async def reject(thread_id: str, body: ReviewRequest, runtime: Runtime) -> TurnResult:
        decision = ReviewDecision(approved=False, reviewer=body.reviewer, note=body.note)
        return await _finish(await runtime.resume(thread_id, decision))

    return app
