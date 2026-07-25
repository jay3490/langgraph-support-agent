from datetime import date
from pathlib import Path

import pytest

from support_agent.cli import main
from support_agent.db import connect, create_demo_database

TODAY = date(2026, 3, 15)


def _count(path: Path, table: str) -> int:
    conn = connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_seed_creates_demo_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "support.db"
    assert create_demo_database(path, today=TODAY) is True
    assert _count(path, "customers") == 4
    assert _count(path, "orders") == 10
    assert _count(path, "refunds") == 1
    assert _count(path, "tickets") == 0


def test_dates_are_anchored_to_today(tmp_path: Path) -> None:
    path = tmp_path / "support.db"
    create_demo_database(path, today=TODAY)
    conn = connect(path)
    try:
        row = conn.execute("SELECT delivered_on FROM orders WHERE id = 'ORD-10010'").fetchone()
    finally:
        conn.close()
    assert row["delivered_on"] == "2026-02-13"  # 30 days before TODAY


def test_existing_database_is_kept_unless_reset(tmp_path: Path) -> None:
    path = tmp_path / "support.db"
    create_demo_database(path, today=TODAY)
    conn = connect(path)
    with conn:
        conn.execute("DELETE FROM refunds")
    conn.close()

    assert create_demo_database(path, today=TODAY) is False
    assert _count(path, "refunds") == 0

    assert create_demo_database(path, today=TODAY, reset=True) is True
    assert _count(path, "refunds") == 1


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "support.db"
    create_demo_database(path, today=TODAY)
    conn = connect(path)
    try:
        with pytest.raises(Exception, match="FOREIGN KEY"):
            conn.execute(
                "INSERT INTO orders (id, customer_id, status, placed_on)"
                " VALUES ('ORD-99999', 'CUST-0000', 'processing', '2026-01-01')"
            )
    finally:
        conn.close()


def test_seed_db_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "cli.db"
    assert main(["seed-db", "--path", str(path)]) == 0
    assert "Seeded demo database" in capsys.readouterr().out
    assert main(["seed-db", "--path", str(path)]) == 0
    assert "--reset" in capsys.readouterr().out
