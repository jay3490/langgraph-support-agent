"""Conversions between decimal amounts and the integer cents stored in the database."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

_CENT = Decimal("0.01")


def to_cents(amount: Decimal | int | str) -> int:
    """Convert an amount in currency units to integer cents, rounding half up."""
    return int((Decimal(amount).quantize(_CENT, rounding=ROUND_HALF_UP) * 100).to_integral_value())


def from_cents(cents: int) -> Decimal:
    return (Decimal(cents) / 100).quantize(_CENT)


def format_cents(cents: int) -> str:
    """Render cents as a plain two-decimal string, e.g. ``8999 -> "89.99"``."""
    return str(from_cents(cents))
