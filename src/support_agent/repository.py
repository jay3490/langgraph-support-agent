"""Data access and business rules for orders, refunds and support tickets.

Each operation opens its own short-lived SQLite connection, so a repository can be
shared freely between threads (graph nodes run tools in worker threads).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from support_agent.db import transaction, utc_now_iso
from support_agent.money import format_cents

TicketCategory = Literal["billing", "technical", "order_status", "escalation", "other"]
TicketPriority = Literal["low", "normal", "high", "urgent"]

AUTO_APPROVER = "auto"
"""``approved_by`` value for refunds that did not need a human reviewer."""


class SupportDataError(Exception):
    """Base class for errors that are safe to show to the customer and the model."""


class NotFoundError(SupportDataError):
    pass


class RefundError(SupportDataError):
    pass


class ApprovalRequiredError(RefundError):
    pass


@dataclass(frozen=True)
class RefundPolicy:
    """Business rules for refunds.

    * Only delivered orders are refundable.
    * A refund must be requested within ``window_days`` of delivery (inclusive);
      gold-tier customers get ``gold_window_days``.
    * Items in ``non_refundable_categories`` are excluded from the refundable amount.
    * Earlier refunds on the same order are subtracted from the refundable amount.
    * Refunds strictly above ``approval_threshold_cents`` need a human reviewer.
    """

    window_days: int = 30
    gold_window_days: int = 45
    approval_threshold_cents: int = 10_000
    non_refundable_categories: frozenset[str] = field(
        default_factory=lambda: frozenset({"digital", "gift_card"})
    )

    def requires_approval(self, amount_cents: int) -> bool:
        return amount_cents > self.approval_threshold_cents


@dataclass(frozen=True)
class Customer:
    id: str
    name: str
    email: str
    phone: str | None
    tier: str


@dataclass(frozen=True)
class OrderItem:
    sku: str
    name: str
    category: str
    quantity: int
    unit_price_cents: int

    @property
    def total_cents(self) -> int:
        return self.quantity * self.unit_price_cents


@dataclass(frozen=True)
class Order:
    id: str
    customer_id: str
    status: str
    placed_on: date
    delivered_on: date | None
    carrier: str | None
    tracking_number: str | None
    items: tuple[OrderItem, ...]
    refunded_cents: int

    @property
    def total_cents(self) -> int:
        return sum(item.total_cents for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.id,
            "customer_id": self.customer_id,
            "status": self.status,
            "placed_on": self.placed_on.isoformat(),
            "delivered_on": self.delivered_on.isoformat() if self.delivered_on else None,
            "carrier": self.carrier,
            "tracking_number": self.tracking_number,
            "total": format_cents(self.total_cents),
            "refunded": format_cents(self.refunded_cents),
            "items": [
                {
                    "sku": item.sku,
                    "name": item.name,
                    "category": item.category,
                    "quantity": item.quantity,
                    "unit_price": format_cents(item.unit_price_cents),
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class RefundEligibility:
    order_id: str
    eligible: bool
    reasons: tuple[str, ...]
    refundable_cents: int
    already_refunded_cents: int
    window_ends: date | None
    requested_cents: int | None
    requires_approval: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "refundable_amount": format_cents(self.refundable_cents),
            "already_refunded": format_cents(self.already_refunded_cents),
            "window_ends": self.window_ends.isoformat() if self.window_ends else None,
            "requested_amount": (
                format_cents(self.requested_cents) if self.requested_cents is not None else None
            ),
            "requires_human_approval": self.requires_approval,
        }


@dataclass(frozen=True)
class Refund:
    id: str
    order_id: str
    amount_cents: int
    reason: str
    approved_by: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "refund_id": self.id,
            "order_id": self.order_id,
            "amount": format_cents(self.amount_cents),
            "reason": self.reason,
            "approved_by": self.approved_by,
        }


@dataclass(frozen=True)
class Ticket:
    id: str
    customer_id: str | None
    order_id: str | None
    category: str
    priority: str
    subject: str
    description: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.id,
            "customer_id": self.customer_id,
            "order_id": self.order_id,
            "category": self.category,
            "priority": self.priority,
            "subject": self.subject,
            "status": self.status,
        }


def _refund_id(row_id: int) -> str:
    return f"REF-{row_id:06d}"


def _ticket_id(row_id: int) -> str:
    return f"TCK-{row_id:06d}"


def _ticket_row_id(ticket_id: str) -> int:
    prefix, _, number = ticket_id.partition("-")
    if prefix != "TCK" or not number.isdigit():
        raise NotFoundError(f"Ticket {ticket_id} was not found.")
    return int(number)


class SupportRepository:
    def __init__(
        self,
        path: Path,
        *,
        policy: RefundPolicy | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.path = path
        self.policy = policy or RefundPolicy()
        self._today = today

    # -- customers and orders -------------------------------------------------------------

    def get_customer(self, customer_id: str) -> Customer:
        with transaction(self.path) as conn:
            row = conn.execute(
                "SELECT id, name, email, phone, tier FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Customer {customer_id} was not found.")
        return Customer(**dict(row))

    def get_order(self, order_id: str) -> Order:
        with transaction(self.path) as conn:
            return self._load_order(conn, order_id)

    def list_orders(self, customer_id: str, *, limit: int = 10) -> list[Order]:
        with transaction(self.path) as conn:
            if (
                conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone()
                is None
            ):
                raise NotFoundError(f"Customer {customer_id} was not found.")
            ids = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM orders WHERE customer_id = ?"
                    " ORDER BY placed_on DESC, id DESC LIMIT ?",
                    (customer_id, limit),
                )
            ]
            return [self._load_order(conn, order_id) for order_id in ids]

    @staticmethod
    def _load_order(conn: sqlite3.Connection, order_id: str) -> Order:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Order {order_id} was not found.")
        items = tuple(
            OrderItem(
                sku=item["sku"],
                name=item["name"],
                category=item["category"],
                quantity=item["quantity"],
                unit_price_cents=item["unit_price_cents"],
            )
            for item in conn.execute(
                "SELECT sku, name, category, quantity, unit_price_cents"
                " FROM order_items WHERE order_id = ? ORDER BY id",
                (order_id,),
            )
        )
        refunded = conn.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) FROM refunds WHERE order_id = ?", (order_id,)
        ).fetchone()[0]
        return Order(
            id=row["id"],
            customer_id=row["customer_id"],
            status=row["status"],
            placed_on=date.fromisoformat(row["placed_on"]),
            delivered_on=date.fromisoformat(row["delivered_on"]) if row["delivered_on"] else None,
            carrier=row["carrier"],
            tracking_number=row["tracking_number"],
            items=items,
            refunded_cents=int(refunded),
        )

    # -- refunds --------------------------------------------------------------------------

    def check_refund_eligibility(
        self, order_id: str, amount_cents: int | None = None
    ) -> RefundEligibility:
        with transaction(self.path) as conn:
            return self._eligibility(conn, order_id, amount_cents)

    def _eligibility(
        self, conn: sqlite3.Connection, order_id: str, amount_cents: int | None
    ) -> RefundEligibility:
        order = self._load_order(conn, order_id)
        tier = conn.execute(
            "SELECT tier FROM customers WHERE id = ?", (order.customer_id,)
        ).fetchone()["tier"]
        policy = self.policy
        reasons: list[str] = []

        window_ends: date | None = None
        if order.status != "delivered":
            reasons.append(
                f"Order status is '{order.status}'; only delivered orders are refundable."
            )
        elif order.delivered_on is not None:
            days = policy.gold_window_days if tier == "gold" else policy.window_days
            window_ends = order.delivered_on + timedelta(days=days)
            if self._today() > window_ends:
                reasons.append(f"The {days}-day refund window ended on {window_ends.isoformat()}.")

        eligible_items_cents = sum(
            item.total_cents
            for item in order.items
            if item.category not in policy.non_refundable_categories
        )
        refundable = max(eligible_items_cents - order.refunded_cents, 0)
        if order.status == "delivered" and eligible_items_cents == 0:
            reasons.append("None of the items on this order are refundable.")
        elif order.status == "delivered" and refundable == 0:
            reasons.append("This order has already been refunded in full.")

        if amount_cents is not None and amount_cents > refundable and not reasons:
            reasons.append(
                f"The requested amount {format_cents(amount_cents)} exceeds the refundable"
                f" amount {format_cents(refundable)}."
            )

        eligible = not reasons
        requested_or_full = amount_cents if amount_cents is not None else refundable
        return RefundEligibility(
            order_id=order.id,
            eligible=eligible,
            reasons=tuple(reasons),
            refundable_cents=refundable if order.status == "delivered" else 0,
            already_refunded_cents=order.refunded_cents,
            window_ends=window_ends,
            requested_cents=amount_cents,
            requires_approval=eligible and policy.requires_approval(requested_or_full),
        )

    def issue_refund(
        self, order_id: str, amount_cents: int, reason: str, *, approved_by: str = AUTO_APPROVER
    ) -> Refund:
        """Record a refund after re-checking every rule inside one transaction.

        Raises :class:`ApprovalRequiredError` when the amount is above the approval
        threshold and no reviewer is named.
        """
        if amount_cents <= 0:
            raise RefundError("The refund amount must be positive.")
        with transaction(self.path) as conn:
            eligibility = self._eligibility(conn, order_id, amount_cents)
            if not eligibility.eligible:
                raise RefundError(" ".join(eligibility.reasons))
            if approved_by == AUTO_APPROVER and self.policy.requires_approval(amount_cents):
                raise ApprovalRequiredError(
                    f"Refunds above {format_cents(self.policy.approval_threshold_cents)}"
                    " need approval from a human reviewer."
                )
            cursor = conn.execute(
                "INSERT INTO refunds (order_id, amount_cents, reason, approved_by, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (order_id, amount_cents, reason, approved_by, utc_now_iso()),
            )
        return Refund(
            id=_refund_id(int(cursor.lastrowid or 0)),
            order_id=order_id,
            amount_cents=amount_cents,
            reason=reason,
            approved_by=approved_by,
        )

    # -- tickets --------------------------------------------------------------------------

    def create_ticket(
        self,
        *,
        subject: str,
        description: str,
        category: TicketCategory,
        priority: TicketPriority = "normal",
        customer_id: str | None = None,
        order_id: str | None = None,
    ) -> Ticket:
        with transaction(self.path) as conn:
            if customer_id is not None and (
                conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone()
                is None
            ):
                raise NotFoundError(f"Customer {customer_id} was not found.")
            if order_id is not None:
                order = conn.execute(
                    "SELECT customer_id FROM orders WHERE id = ?", (order_id,)
                ).fetchone()
                if order is None:
                    raise NotFoundError(f"Order {order_id} was not found.")
                customer_id = customer_id or order["customer_id"]
            cursor = conn.execute(
                "INSERT INTO tickets (customer_id, order_id, category, priority, subject,"
                " description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (customer_id, order_id, category, priority, subject, description, utc_now_iso()),
            )
            row_id = int(cursor.lastrowid or 0)
        return Ticket(
            id=_ticket_id(row_id),
            customer_id=customer_id,
            order_id=order_id,
            category=category,
            priority=priority,
            subject=subject,
            description=description,
            status="open",
        )

    def get_ticket(self, ticket_id: str) -> Ticket:
        with transaction(self.path) as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE id = ?", (_ticket_row_id(ticket_id),)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Ticket {ticket_id} was not found.")
        return Ticket(
            id=_ticket_id(row["id"]),
            customer_id=row["customer_id"],
            order_id=row["order_id"],
            category=row["category"],
            priority=row["priority"],
            subject=row["subject"],
            description=row["description"],
            status=row["status"],
        )
