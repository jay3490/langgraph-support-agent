from decimal import Decimal

import pytest
from pydantic import ValidationError

from support_agent.config import Settings


def test_defaults_need_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPPORT_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.provider == "fake"
    assert settings.openai_api_key is None
    assert settings.refund_approval_threshold == Decimal("100.00")


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORT_PROVIDER", "openai")
    monkeypatch.setenv("SUPPORT_REFUND_APPROVAL_THRESHOLD", "250.50")
    monkeypatch.setenv("SUPPORT_MAX_TOOL_STEPS", "3")
    settings = Settings(_env_file=None)
    assert settings.provider == "openai"
    assert settings.refund_approval_threshold == Decimal("250.50")
    assert settings.max_tool_steps == 3


def test_gold_window_must_not_be_shorter_than_standard() -> None:
    with pytest.raises(ValidationError, match="gold_refund_window_days"):
        Settings(_env_file=None, refund_window_days=30, gold_refund_window_days=10)


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, provider="other")  # type: ignore[arg-type]
