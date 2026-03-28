"""Parser error types and exceptions.

This module defines the exception hierarchy for parsing errors including
XML parsing, JSON parsing, and validation errors.
"""

from enum import Enum


class ParseErrorType(Enum):
    """Types of parsing errors.

    Attributes:
        EMPTY_INPUT: Input content is empty.
        INVALID_XML: XML format is invalid.
        INVALID_JSON: JSON format is invalid.
        MISSING_TOOL_NAME: Required tool name field is missing.
        INVALID_ARGUMENTS: Arguments are invalid.
        UNEXPECTED_FORMAT: Format is not recognized.
        SIZE_LIMIT_EXCEEDED: Content size exceeds limit.
    """
    EMPTY_INPUT = "empty_input"
    INVALID_XML = "invalid_xml"
    INVALID_JSON = "invalid_json"
    MISSING_TOOL_NAME = "missing_tool_name"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNEXPECTED_FORMAT = "unexpected_format"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"


class ParseError(Exception):
    """Base exception for parsing errors.

    Attributes:
        error_type: Type of parsing error.
        message: Human-readable error message.
        context: Optional context string where error occurred.
    """

    def __init__(self, error_type: ParseErrorType, message: str, context: str = "") -> None:
        """Initialize parse error.

        Args:
            error_type: Type of parsing error.
            message: Human-readable error message.
            context: Optional context string where error occurred.
        """
        self.error_type = error_type
        self.message = message
        self.context = context
        super().__init__(f"[{error_type.value}] {message}")

    def __str__(self) -> str:
        """Return string representation of the error.

        Returns:
            Error message with optional context preview.
        """
        if self.context:
            context_preview = self.context[:100]
            return f"{super().__str__()} (context: {context_preview}...)"
        return super().__str__()


class XMLParseError(ParseError):
    """XML parsing error."""

    def __init__(self, message: str, context: str = "") -> None:
        """Initialize XML parse error.

        Args:
            message: Human-readable error message.
            context: Optional context string where error occurred.
        """
        super().__init__(ParseErrorType.INVALID_XML, message, context)


class JSONParseError(ParseError):
    """JSON parsing error."""

    def __init__(self, message: str, context: str = "") -> None:
        """Initialize JSON parse error.

        Args:
            message: Human-readable error message.
            context: Optional context string where error occurred.
        """
        super().__init__(ParseErrorType.INVALID_JSON, message, context)


class ValidationError(ParseError):
    """Validation error."""

    def __init__(
        self,
        message: str,
        context: str = "",
        error_type: ParseErrorType = ParseErrorType.MISSING_TOOL_NAME
    ) -> None:
        """Initialize validation error.

        Args:
            message: Human-readable error message.
            context: Optional context string where error occurred.
            error_type: Type of validation error, defaults to MISSING_TOOL_NAME.
        """
        super().__init__(error_type, message, context)
