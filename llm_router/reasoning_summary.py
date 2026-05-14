"""Small display summaries for provider reasoning items."""

import json
import random
from pathlib import Path

_QUOTES_PATH = Path(__file__).parent / "quotes.json"
_SUMMARY_HEADER = "**少女折寿中**"

with open(_QUOTES_PATH, encoding="utf-8") as _f:
    _QUOTES: tuple[str, ...] = tuple(json.load(_f))


def reasoning_summary_text(
    seed: str | None,
    *,
    will_stop: bool = True,
    rng: random.Random | None = None,
) -> str:
    """Return a short synthetic summary without exposing raw reasoning text."""
    if not will_stop:
        return ""
    rng = rng or random
    quote = rng.choice(_QUOTES)
    return f"{_SUMMARY_HEADER}\n{quote}"
