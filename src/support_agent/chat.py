"""Interactive terminal chat on top of :class:`~support_agent.runtime.SupportRuntime`.

The person at the terminal plays both roles: the customer, and the reviewer who
approves or rejects refunds above the approval threshold.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

from support_agent.approvals import ReviewDecision
from support_agent.runtime import SupportRuntime, TurnEvent

ReadLine = Callable[[str], str]
Write = Callable[[str], None]

QUIT_COMMANDS = frozenset({"/quit", "/exit"})


async def _read(read_line: ReadLine, prompt: str) -> str | None:
    try:
        return await asyncio.to_thread(read_line, prompt)
    except EOFError:
        return None


async def render_turn(events: AsyncIterator[TurnEvent], write: Write) -> dict[str, Any]:
    """Print a turn as it streams and return the data of its ``done`` event."""
    streaming = False
    done: dict[str, Any] = {}
    async for event in events:
        data = event.data
        if event.event == "token":
            if not streaming:
                write("agent> ")
                streaming = True
            write(data["content"])
        elif event.event == "node":
            if data["node"] == "router":
                write(f"[routed to {data['agent']}: {data['reason']}]\n")
            elif "handoff_to" in data:
                write(f"[handed off to {data['handoff_to']}]\n")
            elif data.get("tool_calls"):
                write(f"[{data['node']} calls {', '.join(data['tool_calls'])}]\n")
        elif event.event == "done":
            done = data
    if streaming:
        write("\n")
    elif done.get("reply"):
        write(f"agent> {done['reply']}\n")
    return done


async def ask_review(
    request: dict[str, Any], read_line: ReadLine, write: Write, reviewer: str
) -> ReviewDecision | None:
    """Ask the terminal user to approve the pending refunds; ``None`` means EOF."""
    write(f"Approval needed: refunds above {request['threshold']} USD must be reviewed.\n")
    for refund in request["refunds"]:
        write(f"  {refund['order_id']}: {refund['amount']} USD ({refund['reason']})\n")
    answer = await _read(read_line, "approve? [y/N] ")
    if answer is None:
        return None
    approved = answer.strip().lower() in {"y", "yes"}
    note = await _read(read_line, "note (optional)> ")
    return ReviewDecision(approved=approved, reviewer=reviewer, note=(note or "").strip() or None)


async def run_chat(
    runtime: SupportRuntime,
    *,
    read_line: ReadLine,
    write: Write,
    customer_id: str | None = None,
    thread_id: str | None = None,
    reviewer: str = "terminal",
) -> str:
    """Run the chat loop until ``/quit`` or end of input; returns the thread id."""
    if thread_id is None:
        thread_id = await runtime.create_thread(customer_id)
        pending = None
    else:
        pending = (await runtime.get_thread(thread_id)).pending_approval
    write(f"Thread {thread_id}. Type /quit to leave.\n")

    while True:
        if pending is not None:
            decision = await ask_review(pending, read_line, write, reviewer)
            if decision is None:
                break
            result = await render_turn(await runtime.resume(thread_id, decision), write)
            pending = result.get("pending_approval")
            continue

        line = await _read(read_line, "you> ")
        if line is None or line.strip() in QUIT_COMMANDS:
            break
        if not line.strip():
            continue
        result = await render_turn(await runtime.send_message(thread_id, line), write)
        pending = result.get("pending_approval")

    write(f"Bye. Resume this conversation with --thread {thread_id}\n")
    return thread_id
