"""Parser base classes and data structures."""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# Maximum content size for parsing (10MB)
MAX_CONTENT_SIZE = 10 * 1024 * 1024


def validate_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tool_schema: dict | None
) -> list[str]:
    """
    Validate tool arguments against schema.

    Performs basic validation:
    - Checks required fields are present
    - Checks for unexpected fields if additionalProperties is false
    - Does NOT validate nested schemas or complex types

    Args:
        tool_name: Name of the tool (for error messages)
        arguments: Parsed arguments to validate
        tool_schema: Tool schema in OpenAI format

    Returns:
        List of validation error messages (empty if valid)
    """
    if not tool_schema:
        # No schema available, skip validation
        return []

    errors = []

    # Extract parameter schema
    params = tool_schema.get("function", {}).get("parameters", {})
    if not params:
        return []

    properties = params.get("properties", {})
    required = params.get("required", [])
    additional_props = params.get("additionalProperties", True)

    # Check required fields
    for field in required:
        if field not in arguments:
            errors.append(f"Missing required parameter '{field}' for tool '{tool_name}'")

    # Check for unexpected fields if additionalProperties is false
    if not additional_props and properties:
        for field in arguments:
            if field not in properties:
                errors.append(f"Unexpected parameter '{field}' for tool '{tool_name}'")

    return errors


@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    tool_name: str
    arguments: dict[str, Any]
    server_name: str = "tools"

    def to_openai_format(self) -> dict:
        """Convert to OpenAI tool_calls format."""
        return {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False)
            }
        }


class ToolNameResolver:
    """Resolve tool names by combining server_name and tool_name."""

    def __init__(self):
        # Cache: (server_name, tool_name) -> resolved_name
        self._cache: dict[tuple[str, str], str] = {}

    def resolve(
        self,
        server_name: str,
        tool_name: str,
        available_tools: list[dict] | None = None
    ) -> str:
        """
        Resolve the actual tool name by combining server_name and tool_name.

        Args:
            server_name: Server name from MCP XML
            tool_name: Tool name from MCP XML
            available_tools: List of available tools in OpenAI format

        Returns:
            Resolved tool name
        """
        # Default server: return tool_name as-is
        if not server_name or server_name == "default" or server_name == "tools":
            return tool_name

        # Check cache
        cache_key = (server_name, tool_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # No tools list: return tool_name
        if not available_tools:
            return tool_name

        # Find matching tool in available tools
        candidates = []
        for tool in available_tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                name = func.get("name", "")

                # Check if tool_name is in the name
                if tool_name in name:
                    candidates.append(name)

        # Single candidate: use it
        if len(candidates) == 1:
            resolved = candidates[0]
            self._cache[cache_key] = resolved
            return resolved

        # Multiple candidates: find one containing server_name
        for candidate in candidates:
            if server_name in candidate:
                self._cache[cache_key] = candidate
                return candidate

        # No match: return tool_name
        return tool_name

    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing tool calls."""
    success: bool
    tool_calls: list[ToolCall]
    errors: list[str]
    warnings: list[str]

    @classmethod
    def ok(cls, tool_calls: list[ToolCall], warnings: list[str] | None = None) -> ParseResult:
        """Create successful parse result."""
        return cls(
            success=True,
            tool_calls=tool_calls,
            errors=[],
            warnings=warnings or []
        )

    @classmethod
    def error(cls, errors: list[str], warnings: list[str] | None = None) -> ParseResult:
        """Create failed parse result."""
        return cls(
            success=False,
            tool_calls=[],
            errors=errors,
            warnings=warnings or []
        )


class ToolCallParser(ABC):
    """Abstract base class for tool call parsers."""

    @abstractmethod
    def parse(self, content: str) -> ParseResult:
        """Parse content and return result with all found tool calls."""
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Parser format name for debugging."""
        pass
