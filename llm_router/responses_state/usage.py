"""Usage normalization for Responses-compatible replies."""

from __future__ import annotations

from typing import Any


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_cached_input_tokens(raw_usage: dict[str, Any]) -> int:
    for details_key in ("input_tokens_details", "prompt_tokens_details"):
        details = raw_usage.get(details_key)
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
            if cached_tokens is not None:
                return _int_or_zero(cached_tokens)

    for usage_key in ("prompt_cache_hit_tokens", "cached_input_tokens"):
        cached_tokens = raw_usage.get(usage_key)
        if cached_tokens is not None:
            return _int_or_zero(cached_tokens)

    return 0


def _extract_reasoning_tokens(raw_usage: dict[str, Any]) -> int:
    details = raw_usage.get("output_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            return _int_or_zero(reasoning_tokens)

    for usage_key in ("reasoning_tokens", "reasoning_output_tokens"):
        reasoning_tokens = raw_usage.get(usage_key)
        if reasoning_tokens is not None:
            return _int_or_zero(reasoning_tokens)

    return 0


def _responses_usage_from_provider(raw_usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _int_or_zero(
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)),
    )
    output_tokens = _int_or_zero(
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)),
    )
    total_tokens = _int_or_zero(
        raw_usage.get("total_tokens", input_tokens + output_tokens),
    )

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": _extract_cached_input_tokens(raw_usage),
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": _extract_reasoning_tokens(raw_usage),
        },
        "total_tokens": total_tokens,
    }
