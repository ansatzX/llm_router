"""Small display summaries for provider reasoning items."""

from __future__ import annotations

import hashlib

REASONING_SUMMARY_QUOTES = (
    "少女折寿中",
    "我思故我在",
    "大胆假设，小心求证",
    "知其然，知其所以然",
    "慢慢想，比较快",
)


def reasoning_summary_text(seed: str | None) -> str:
    """Return a short synthetic summary without exposing raw reasoning text."""
    if not seed:
        return REASONING_SUMMARY_QUOTES[0]
    digest = hashlib.blake2s(seed.encode("utf-8"), digest_size=2).digest()
    index = int.from_bytes(digest, "big") % len(REASONING_SUMMARY_QUOTES)
    return REASONING_SUMMARY_QUOTES[index]
