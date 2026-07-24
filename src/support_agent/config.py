"""Typed settings loaded from ``SUPPORT_*`` environment variables (or a ``.env`` file)."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openai"]


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden with an environment variable named after it with a
    ``SUPPORT_`` prefix, e.g. ``SUPPORT_PROVIDER=openai``.
    """

    model_config = SettingsConfigDict(env_prefix="SUPPORT_", env_file=".env", extra="ignore")

    # LLM provider. ``fake`` is scripted and needs no credentials; ``openai`` talks to any
    # OpenAI-compatible endpoint (OpenAI itself, or a local Ollama server via its /v1 API).
    provider: Provider = "fake"
    model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_api_key: SecretStr | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Storage.
    database_path: Path = Path("var/support.db")
    checkpoint_path: Path = Path("var/checkpoints.db")
    kb_dir: Path = Path("kb")
    seed_on_startup: bool = True

    # Business rules.
    refund_approval_threshold: Decimal = Field(default=Decimal("100.00"), ge=0, decimal_places=2)
    refund_window_days: int = Field(default=30, ge=0)
    gold_refund_window_days: int = Field(default=45, ge=0)

    # Guardrails.
    max_tool_steps: int = Field(default=8, ge=1)
    max_handoffs: int = Field(default=2, ge=0)

    # HTTP server and logging.
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _gold_window_not_shorter(self) -> Settings:
        if self.gold_refund_window_days < self.refund_window_days:
            raise ValueError("gold_refund_window_days must be >= refund_window_days")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read once from the environment."""
    return Settings()
