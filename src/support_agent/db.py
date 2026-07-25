"""SQLite schema and the demo data set the support tools run against.

Money is stored as integer cents. Dates are ISO-8601 strings; ``created_at`` columns
hold UTC timestamps. The demo data is anchored to a ``today`` date so that refund
windows stay meaningful whenever the database is seeded.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id    TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    tier  TEXT NOT NULL CHECK (tier IN ('standard', 'gold'))
);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers (id),
    status          TEXT NOT NULL
                    CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled')),
    placed_on       TEXT NOT NULL,
    delivered_on    TEXT,
    carrier         TEXT,
    tracking_number TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id         TEXT NOT NULL REFERENCES orders (id),
    sku              TEXT NOT NULL,
    name             TEXT NOT NULL,
    category         TEXT NOT NULL,
    quantity         INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0)
);

CREATE TABLE IF NOT EXISTS refunds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL REFERENCES orders (id),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    reason       TEXT NOT NULL,
    approved_by  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT REFERENCES customers (id),
    order_id    TEXT REFERENCES orders (id),
    category    TEXT NOT NULL,
    priority    TEXT NOT NULL,
    subject     TEXT NOT NULL,
    description TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_items_order ON order_items (order_id);
CREATE INDEX IF NOT EXISTS idx_refunds_order ON refunds (order_id);
"""


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and rows addressable by name."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a connection that commits on success, rolls back on error and always closes."""
    conn = connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@dataclass(frozen=True)
class _Item:
    sku: str
    name: str
    category: str
    unit_price_cents: int
    quantity: int = 1


@dataclass(frozen=True)
class _Order:
    id: str
    customer_id: str
    status: str
    placed_days_ago: int
    items: tuple[_Item, ...]
    delivered_days_ago: int | None = None
    carrier: str | None = None
    tracking_number: str | None = None


_CUSTOMERS = (
    ("CUST-1001", "Alice Nguyen", "alice.nguyen@example.com", "+1-555-010-1001", "standard"),
    ("CUST-1002", "Bram Okafor", "bram.okafor@example.com", "+1-555-010-1002", "gold"),
    ("CUST-1003", "Chen Li", "chen.li@example.com", "+1-555-010-1003", "standard"),
    ("CUST-1004", "Dana Petrova", "dana.petrova@example.com", "+1-555-010-1004", "gold"),
)

_ORDERS = (
    # Delivered recently, below the approval threshold: refundable without review.
    _Order(
        "ORD-10001",
        "CUST-1001",
        "delivered",
        14,
        (_Item("AUD-200", "Wireless headphones", "electronics", 8999),),
        delivered_days_ago=10,
        carrier="UPS",
        tracking_number="1Z999AA10000000001",
    ),
    # Delivered recently, above the approval threshold: a refund needs human approval.
    _Order(
        "ORD-10002",
        "CUST-1001",
        "delivered",
        16,
        (_Item("FUR-510", "Standing desk", "furniture", 34900),),
        delivered_days_ago=12,
        carrier="FedEx",
        tracking_number="7489000000000002",
    ),
    # Still in transit.
    _Order(
        "ORD-10003",
        "CUST-1001",
        "shipped",
        3,
        (_Item("ACC-110", "USB-C cable", "accessories", 1299, 2),),
        carrier="UPS",
        tracking_number="1Z999AA10000000003",
    ),
    # Gold customer, 40 days after delivery: inside the extended gold window only.
    _Order(
        "ORD-10004",
        "CUST-1002",
        "delivered",
        44,
        (_Item("MON-270", "27-inch monitor", "electronics", 22900),),
        delivered_days_ago=40,
        carrier="DHL",
        tracking_number="JD000000000000004",
    ),
    # Standard customer, 40 days after delivery: outside the refund window.
    _Order(
        "ORD-10005",
        "CUST-1003",
        "delivered",
        45,
        (_Item("KIT-330", "Espresso grinder", "kitchen", 15900),),
        delivered_days_ago=40,
        carrier="UPS",
        tracking_number="1Z999AA10000000005",
    ),
    # Not shipped yet.
    _Order(
        "ORD-10006",
        "CUST-1003",
        "processing",
        1,
        (_Item("BOK-001", "Field guide to birds", "books", 2450),),
    ),
    # Mixed basket: digital goods and gift cards are not refundable.
    _Order(
        "ORD-10007",
        "CUST-1004",
        "delivered",
        7,
        (
            _Item("ACC-410", "Laptop sleeve", "accessories", 3900),
            _Item("GFT-050", "Gift card", "gift_card", 5000),
            _Item("SFT-120", "Photo editor licence", "digital", 12000),
        ),
        delivered_days_ago=5,
        carrier="USPS",
        tracking_number="9400100000000000000007",
    ),
    # Cancelled before shipping.
    _Order(
        "ORD-10008",
        "CUST-1004",
        "cancelled",
        20,
        (_Item("CAM-900", "Action camera", "electronics", 27900),),
    ),
    # Partially refunded already (see _REFUNDS).
    _Order(
        "ORD-10009",
        "CUST-1002",
        "delivered",
        6,
        (_Item("KEY-129", "Mechanical keyboard", "electronics", 12900),),
        delivered_days_ago=3,
        carrier="UPS",
        tracking_number="1Z999AA10000000009",
    ),
    # Delivered exactly on the edge of the standard 30-day window.
    _Order(
        "ORD-10010",
        "CUST-1003",
        "delivered",
        34,
        (_Item("LMP-050", "Desk lamp", "home", 4500),),
        delivered_days_ago=30,
        carrier="FedEx",
        tracking_number="7489000000000010",
    ),
)

_REFUNDS = (("ORD-10009", 3000, "Price adjustment after promotion", "billing-team"),)


def seed_demo_data(conn: sqlite3.Connection, *, today: date) -> None:
    """Insert the demo customers, orders and refunds, anchored to ``today``."""
    conn.executemany(
        "INSERT INTO customers (id, name, email, phone, tier) VALUES (?, ?, ?, ?, ?)",
        _CUSTOMERS,
    )
    for order in _ORDERS:
        delivered = (
            None
            if order.delivered_days_ago is None
            else (today - timedelta(days=order.delivered_days_ago)).isoformat()
        )
        conn.execute(
            "INSERT INTO orders (id, customer_id, status, placed_on, delivered_on, carrier,"
            " tracking_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                order.id,
                order.customer_id,
                order.status,
                (today - timedelta(days=order.placed_days_ago)).isoformat(),
                delivered,
                order.carrier,
                order.tracking_number,
            ),
        )
        conn.executemany(
            "INSERT INTO order_items (order_id, sku, name, category, quantity, unit_price_cents)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (order.id, i.sku, i.name, i.category, i.quantity, i.unit_price_cents)
                for i in order.items
            ],
        )
    conn.executemany(
        "INSERT INTO refunds (order_id, amount_cents, reason, approved_by, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        [(*refund, utc_now_iso()) for refund in _REFUNDS],
    )


def create_demo_database(path: Path, *, today: date | None = None, reset: bool = False) -> bool:
    """Create and seed the demo database at ``path``.

    Returns ``True`` when the database was (re)created and ``False`` when it already
    existed and ``reset`` was not requested.
    """
    if path.exists():
        if not reset:
            return False
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with transaction(path) as conn:
        init_schema(conn)
        seed_demo_data(conn, today=today or date.today())
    return True
