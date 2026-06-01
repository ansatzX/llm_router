"""Client-visible provider error mapping helpers."""

from __future__ import annotations

from typing import Any

from llm_router.llm_client import LLMRequestError


def _deepseek_missing_reasoning_tool_call_ids(
    payload: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(payload, dict):
        return []

    missing: list[str] = []
    for message in payload.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls") or []
        if (
            message.get("role") != "assistant"
            or not isinstance(tool_calls, list)
            or message.get("reasoning_content")
        ):
            continue
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str):
                missing.append(tool_call["id"])
    return missing


def _is_deepseek_missing_reasoning_error(error: LLMRequestError) -> bool:
    message = error.message.lower()
    return (
        ("reasoning_content" in message or "content[].thinking" in message)
        and "thinking mode" in message
    )


def _llm_request_error_body(
    error: LLMRequestError,
    *,
    payload: dict[str, Any] | None = None,
    is_deepseek: bool = False,
) -> tuple[dict[str, Any], int]:
    if is_deepseek and _is_deepseek_missing_reasoning_error(error):
        missing_call_ids = _deepseek_missing_reasoning_tool_call_ids(payload)
        message = (
            "DeepSeek rejected this continuation because thinking-mode tool "
            "replay requires the original provider reasoning_content."
        )
        if missing_call_ids:
            message += (
                " Missing reasoning_content for tool call(s): "
                f"{', '.join(missing_call_ids)}."
            )
        message += f" Provider message: {error.message}"
        return {
            "error": {
                "type": "invalid_request_error",
                "code": "deepseek_missing_reasoning_content",
                "message": message,
                "provider_status": error.status_code,
            }
        }, 409

    status_code = error.status_code if 400 <= error.status_code < 500 else 502
    return {
        "error": {
            "type": "provider_error",
            "code": "provider_error",
            "message": error.message,
            "provider_status": error.status_code,
        }
    }, status_code
