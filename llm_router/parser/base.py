"""Parser base classes and data structures."""
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    tool_name: str
    arguments: dict
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


@dataclass
class ParseResult:
    """Result of parsing tool calls."""
    success: bool
    tool_calls: list[ToolCall]
    errors: list[str]
    warnings: list[str]

    @classmethod
    def ok(cls, tool_calls: list[ToolCall], warnings: list[str] = None) -> 'ParseResult':
        """Create successful parse result."""
        return cls(
            success=True,
            tool_calls=tool_calls,
            errors=[],
            warnings=warnings or []
        )

    @classmethod
    def error(cls, errors: list[str], warnings: list[str] = None) -> 'ParseResult':
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
