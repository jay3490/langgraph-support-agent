from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from support_agent import cli
from support_agent.chat import run_chat
from support_agent.config import Settings, get_settings
from support_agent.demo_policy import DemoPolicy
from support_agent.llm import FakeChatModel
from support_agent.runtime import SupportRuntime, open_runtime
from tests.conftest import KB_DIR, TODAY


class Terminal:
    """Scripted stdin and captured stdout."""

    def __init__(self, *lines: str) -> None:
        self._lines: Iterator[str] = iter(lines)
        self.prompts: list[str] = []
        self.output: list[str] = []

    def read_line(self, prompt: str) -> str:
        self.prompts.append(prompt)
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None

    def write(self, text: str) -> None:
        self.output.append(text)

    @property
    def text(self) -> str:
        return "".join(self.output)


@pytest.fixture
async def runtime(settings: Settings):
    model = FakeChatModel(responder=DemoPolicy())
    async with open_runtime(settings, model=model, today=TODAY) as rt:
        yield rt


async def chat(runtime: SupportRuntime, terminal: Terminal, **kwargs: object) -> str:
    return await run_chat(
        runtime,
        read_line=terminal.read_line,
        write=terminal.write,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_conversation_streams_replies(runtime: SupportRuntime) -> None:
    terminal = Terminal("Where is ORD-10003?", "", "/quit")
    thread = await chat(runtime, terminal, customer_id="CUST-1001")
    text = terminal.text
    assert f"Thread {thread}" in text
    assert "[routed to order_status" in text
    assert "[order_status calls lookup_order]" in text
    assert "agent> Order ORD-10003 (USB-C cable) is on its way with UPS." in text
    assert terminal.prompts == ["you> ", "you> ", "you> "]
    assert text.endswith(f"--thread {thread}\n")


@pytest.mark.anyio
async def test_terminal_user_approves_refund(runtime: SupportRuntime) -> None:
    terminal = Terminal("Refund ORD-10002", "y", "photos checked", "/quit")
    await chat(runtime, terminal, customer_id="CUST-1001", reviewer="dana")
    assert "Approval needed: refunds above 100.00 USD" in terminal.text
    assert "ORD-10002: 349.00 USD" in terminal.text
    assert "refunded 349.00 USD" in terminal.text
    assert runtime.repository.get_order("ORD-10002").refunded_cents == 34900


@pytest.mark.anyio
async def test_terminal_user_rejects_refund(runtime: SupportRuntime) -> None:
    terminal = Terminal("Refund ORD-10002", "n", "", "/quit")
    await chat(runtime, terminal, customer_id="CUST-1001")
    assert "wasn't able to issue the refund" in terminal.text
    assert "rejected by reviewer terminal." in terminal.text
    assert runtime.repository.get_order("ORD-10002").refunded_cents == 0


@pytest.mark.anyio
async def test_resume_thread_waiting_for_approval(runtime: SupportRuntime) -> None:
    first = Terminal("Refund ORD-10002")  # input ends while the refund awaits review
    thread = await chat(runtime, first, customer_id="CUST-1001")
    assert "approve? [y/N] " in first.prompts

    second = Terminal("yes", "", "/exit")
    assert await chat(runtime, second, thread_id=thread) == thread
    assert second.prompts[0] == "approve? [y/N] "
    assert "refunded 349.00 USD" in second.text


def test_chat_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SUPPORT_DATABASE_PATH", str(tmp_path / "support.db"))
    monkeypatch.setenv("SUPPORT_CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setenv("SUPPORT_KB_DIR", str(KB_DIR))
    monkeypatch.setenv("SUPPORT_PROVIDER", "fake")
    get_settings.cache_clear()
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    replies = iter(["I forgot my password", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    try:
        assert cli.main(["chat", "--customer", "CUST-1001"]) == 0
        out = capsys.readouterr().out
        assert '"Resetting your password"' in out

        assert cli.main(["chat", "--thread", "does-not-exist"]) == 1
        assert "was not found" in capsys.readouterr().err
    finally:
        get_settings.cache_clear()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
