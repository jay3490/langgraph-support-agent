from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from support_agent.db import create_demo_database
from support_agent.repository import RefundPolicy, SupportRepository

TODAY = date(2026, 3, 15)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "support.db"
    create_demo_database(path, today=TODAY)
    return path


@pytest.fixture
def repository(db_path: Path) -> SupportRepository:
    return SupportRepository(db_path, policy=RefundPolicy(), today=lambda: TODAY)
