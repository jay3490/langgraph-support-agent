"""Logging setup with personal data masked before anything is written.

Customer messages and tool arguments are logged for debugging, and they can contain
email addresses, phone numbers or card numbers. :class:`PiiMaskingFilter` rewrites every
record that reaches the handler, including exception text, so those values never reach
the log output.
"""

from __future__ import annotations

import logging
import re

_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}(\d{4})(?![\w-])")
_PHONE = re.compile(r"(?<![\w+(])(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?(\d{4})(?!\w)")
_EMAIL = re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)\b")


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if not 13 <= len(digits) <= 19:
        return match.group(0)
    return f"[card ****{match.group(1)}]"


def mask_pii(text: str) -> str:
    """Mask card numbers (keeping the last four digits), phone numbers and emails."""
    text = _CARD.sub(_mask_card, text)
    text = _PHONE.sub(lambda m: f"[phone ***{m.group(1)[-2:]}]", text)
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


class PiiMaskingFilter(logging.Filter):
    """Replace each record's message (and exception text) with a masked copy."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_pii(record.getMessage())
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = mask_pii(record.exc_text)
        return True


LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> logging.Handler:
    """Send all log records to stderr through the masking filter.

    Replaces the handlers on the root logger, so call it once from an entry point.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(PiiMaskingFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Keep third-party chatter at WARNING unless debugging.
    for noisy in ("httpx", "httpcore", "openai", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING if level != "DEBUG" else logging.DEBUG)
    return handler
