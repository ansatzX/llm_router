"""Small display summaries for provider reasoning items."""

from __future__ import annotations

import json
import random
from pathlib import Path

_QUOTES_PATH = Path(__file__).parent / "quotes.json"

with open(_QUOTES_PATH, encoding="utf-8") as _f:
    _QUOTES: tuple[str, ...] = tuple(json.load(_f))


def reasoning_summary_text(seed: str | None) -> str:
    """Return a short synthetic summary without exposing raw reasoning text."""
    if not seed:
        return random.choice(_QUOTES)
    digest = int.from_bytes(seed.encode("utf-8")[:4], "big")
    return _QUOTES[digest % len(_QUOTES)]
