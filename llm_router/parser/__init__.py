"""
Parser module for extracting tool calls from LLM responses.

Public API:
- parse_tool_calls(): Main entry point for parsing
- ToolCall: Data class for tool call information
- ParseResult: Result container with tool_calls/errors/warnings
- ParseError: Base exception for parsing errors
"""

from .base import ParseResult, ToolCall
from .errors import ParseError

__all__ = [
    'ToolCall',
    'ParseResult',
    'ParseError',
    'parse_tool_calls',
]


def parse_tool_calls(
    content: str,
    reasoning_content: str = "",
    prefer_format: str = None
) -> ParseResult:
    """
    Parse tool calls from LLM response content.

    Args:
        content: Main response content
        reasoning_content: Reasoning content (MiroThinker may put tool calls here)
        prefer_format: Optional hint - "xml" or "json" to try that parser first

    Returns:
        ParseResult with success status, tool_calls, errors, and warnings

    Note:
        - Content and reasoning_content are merged before parsing
        - Returns first successful parser result (partial success is success)
        - Errors/warnings aggregated from failed parsers if all fail
    """
    from .json_parser import JSONParser
    from .xml_parser import XMLParser

    # Merge content
    full_content = f"{content}\n{reasoning_content}" if reasoning_content else content

    # Determine parser order
    if prefer_format == "json":
        parsers = [JSONParser(), XMLParser()]
    else:  # Default: XML first
        parsers = [XMLParser(), JSONParser()]

    # Try each parser
    all_errors = []
    all_warnings = []

    for parser in parsers:
        result = parser.parse(full_content)

        # If parser found any tool calls, return immediately
        if result.tool_calls:
            return result

        # Otherwise, collect errors and try next parser
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)

    # All parsers failed to find tool calls
    return ParseResult.error(
        errors=all_errors or ["No tool calls found in any format"],
        warnings=all_warnings
    )
