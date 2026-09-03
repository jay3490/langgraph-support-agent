from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest
from langchain_core.messages import ToolCall

from support_agent.logging_config import PiiMaskingFilter, configure_logging, mask_pii
from support_agent.tools import CREATE_SUPPORT_TICKET, ToolContext, Toolkit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail alice.nguyen@example.com now", "mail a***@example.com now"),
        ("x@sub.example.co.uk", "x***@sub.example.co.uk"),
        ("card 4111 1111 1111 1111 ok", "card [card ****1111] ok"),
        ("card 4111-1111-1111-1111", "card [card ****1111]"),
        ("card 4111111111111111", "card [card ****1111]"),
        ("call +1-555-010-1001", "call [phone ***01]"),
        ("call (555) 010-1001 today", "call [phone ***01] today"),
        ("call 555.010.1001", "call [phone ***01]"),
    ],
)
def test_mask_pii(text: str, expected: str) -> None:
    assert mask_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Order ORD-10001 refund REF-000002 ticket TCK-000001",
        "tracking 1Z999AA10000000001 and 9400100000000000000007",
        "amount 349.00 on 2026-03-15",
        "customer CUST-1001",
    ],
)
def test_business_identifiers_are_left_alone(text: str) -> None:
    assert mask_pii(text) == text


@pytest.fixture
def captured() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(PiiMaskingFilter())
    logger = logging.getLogger("support_agent")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def test_filter_masks_arguments_and_exceptions(captured: io.StringIO) -> None:
    logger = logging.getLogger("support_agent.test")
    logger.info("customer %s wrote from %s", "CUST-1001", "dana.petrova@example.com")
    try:
        raise ValueError("bad card 5500 0000 0000 0004")
    except ValueError:
        logger.exception("failed for bram.okafor@example.com")
    output = captured.getvalue()
    assert "dana.petrova@example.com" not in output
    assert "d***@example.com" in output
    assert "b***@example.com" in output
    assert "5500 0000 0000 0004" not in output
    assert "[card ****0004]" in output


def test_tool_call_logs_are_masked(toolkit: Toolkit, captured: io.StringIO) -> None:
    call = ToolCall(
        name=CREATE_SUPPORT_TICKET,
        args={
            "subject": "Call me back",
            "description": "Please call +1-555-010-1004 or write to dana.petrova@example.com",
            "category": "other",
        },
        id="t1",
    )
    assert toolkit.execute(call, ToolContext(customer_id="CUST-1004")).status == "success"
    output = captured.getvalue()
    assert "Calling create_support_ticket" in output
    assert "dana.petrova@example.com" not in output
    assert "555-010-1004" not in output


def test_configure_logging_installs_the_filter() -> None:
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        handler = configure_logging("DEBUG")
        assert root.handlers == [handler]
        assert any(isinstance(f, PiiMaskingFilter) for f in handler.filters)
        assert root.level == logging.DEBUG
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
