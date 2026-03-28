"""Tool call validation functions.

This module provides validation utilities for checking tool call
completeness and sanitizing arguments.
"""

from typing import Any

from llm_router.parser.base import ToolCall


def validate_tool_call(tool_call: ToolCall) -> tuple[bool, str | None]:
    """Validate tool call data completeness.

    Args:
        tool_call: Tool call instance to validate.

    Returns:
        Tuple of (is_valid, warning_message) where:
        - is_valid: True if tool call can be used.
        - warning_message: Warning if valid but has issues, error if invalid.
    """
    warnings: list[str] = []

    # Check tool_name
    if not tool_call.tool_name:
        return False, "Tool name is empty"

    if not isinstance(tool_call.tool_name, str):
        return False, f"Tool name must be string, got {type(tool_call.tool_name).__name__}"

    if not tool_call.tool_name.strip():
        return False, "Tool name is whitespace only"

    # Check arguments
    if not isinstance(tool_call.arguments, dict):
        return False, f"Arguments must be dict, got {type(tool_call.arguments).__name__}"

    # Check for None values in arguments (warning, not error)
    if tool_call.arguments:
        none_keys = [k for k, v in tool_call.arguments.items() if v is None]
        if none_keys:
            warnings.append(f"Arguments contain None values for keys: {none_keys}")

    # Return result
    warning_msg = "; ".join(warnings) if warnings else None
    return True, warning_msg


def sanitize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Clean arguments by removing None values.

    Args:
        arguments: Raw arguments dictionary.

    Returns:
        Cleaned arguments dictionary with None values removed.
    """
    if not isinstance(arguments, dict):
        return {}

    return {k: v for k, v in arguments.items() if v is not None}
