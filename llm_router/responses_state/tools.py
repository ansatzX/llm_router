"""Tool schema helpers for Responses and Chat compatibility."""

from __future__ import annotations

from typing import Any

from llm_router.deepseek import DeepSeekChatAdapter


def convert_chat_tool_to_responses(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Chat Completions tool format to Responses API format."""
    if tool.get("type") == "function":
        if "function" in tool:
            func = tool["function"]
            parameters = func.get("parameters")
            if parameters is None:
                parameters = func.get("input_schema", {})
            return {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": parameters,
            }
        if "name" in tool:
            normalized = dict(tool)
            if "parameters" not in normalized and "input_schema" in normalized:
                normalized["parameters"] = normalized["input_schema"]
            normalized.pop("input_schema", None)
            return normalized
    return tool


def convert_responses_tool_to_chat(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses API function tool format to Chat Completions format."""
    return DeepSeekChatAdapter().response_tool_to_chat(tool)


def _normalize_responses_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Normalize mixed tool schema aliases to the Responses function shape."""
    if not isinstance(tools, list):
        return []
    return [
        convert_chat_tool_to_responses(tool)
        for tool in tools
        if isinstance(tool, dict)
    ]
