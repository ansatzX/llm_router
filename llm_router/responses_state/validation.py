"""Validation helpers for Responses conversation state."""

from __future__ import annotations

from typing import Any

TOOL_CALL_TYPES = {"function_call", "custom_tool_call"}
TOOL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}


class ResponsesStateError(ValueError):
    """Client-visible Responses state-machine error."""

    def __init__(self, message: str, code: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code

    def to_error_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": self.code,
            },
        }


def item_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or "")


def is_tool_call(item: dict[str, Any]) -> bool:
    return item_type(item) in TOOL_CALL_TYPES


def is_tool_output(item: dict[str, Any]) -> bool:
    return item_type(item) in TOOL_OUTPUT_TYPES


def tool_call_id(item: dict[str, Any]) -> str:
    return str(item.get("call_id") or item.get("id") or "")


def validate_tool_output_known(
    call_id: str,
    unresolved_call_ids: set[str],
    known_call_ids: set[str],
) -> None:
    if not call_id:
        raise ResponsesStateError(
            "Tool output item is missing call_id.",
            "invalid_tool_output",
        )
    if call_id not in known_call_ids:
        raise ResponsesStateError(
            f"Tool output references unknown call_id: {call_id}",
            "invalid_tool_output",
        )
    if call_id not in unresolved_call_ids:
        raise ResponsesStateError(
            f"Tool output references an already satisfied call_id: {call_id}",
            "invalid_tool_output",
        )


def validate_no_pending_before_client_message(
    unresolved_call_ids: set[str],
) -> None:
    if unresolved_call_ids:
        pending = ", ".join(sorted(unresolved_call_ids))
        raise ResponsesStateError(
            "Pending tool calls must be satisfied before continuing: "
            f"{pending}",
            "pending_tool_calls",
        )
