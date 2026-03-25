"""Parser error types and exceptions."""
from enum import Enum


class ParseErrorType(Enum):
    """Types of parsing errors."""
    EMPTY_INPUT = "empty_input"
    INVALID_XML = "invalid_xml"
    INVALID_JSON = "invalid_json"
    MISSING_TOOL_NAME = "missing_tool_name"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNEXPECTED_FORMAT = "unexpected_format"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"


class ParseError(Exception):
    """Base exception for parsing errors."""

    def __init__(self, error_type: ParseErrorType, message: str, context: str = ""):
        self.error_type = error_type
        self.message = message
        self.context = context
        super().__init__(f"[{error_type.value}] {message}")

    def __str__(self):
        context_preview = self.context[:100] if self.context else ""
        return f"{super().__str__()} (context: {context_preview}...)"


class XMLParseError(ParseError):
    """XML parsing error."""

    def __init__(self, message: str, context: str = ""):
        super().__init__(ParseErrorType.INVALID_XML, message, context)


class JSONParseError(ParseError):
    """JSON parsing error."""

    def __init__(self, message: str, context: str = ""):
        super().__init__(ParseErrorType.INVALID_JSON, message, context)


class ValidationError(ParseError):
    """Validation error."""

    def __init__(self, message: str, context: str = ""):
        super().__init__(ParseErrorType.MISSING_TOOL_NAME, message, context)
