"""Tests for parser error types."""
from llm_router.parser.errors import (
    JSONParseError,
    ParseError,
    ParseErrorType,
    ValidationError,
    XMLParseError,
)


def test_parse_error_type_enum():
    """Test ParseErrorType enum values."""
    assert ParseErrorType.EMPTY_INPUT.value == "empty_input"
    assert ParseErrorType.INVALID_XML.value == "invalid_xml"
    assert ParseErrorType.INVALID_JSON.value == "invalid_json"
    assert ParseErrorType.MISSING_TOOL_NAME.value == "missing_tool_name"
    assert ParseErrorType.INVALID_ARGUMENTS.value == "invalid_arguments"
    assert ParseErrorType.UNEXPECTED_FORMAT.value == "unexpected_format"
    assert ParseErrorType.SIZE_LIMIT_EXCEEDED.value == "size_limit_exceeded"


def test_parse_error_creation():
    """Test ParseError instantiation."""
    error = ParseError(
        ParseErrorType.INVALID_XML,
        "Test message",
        "context snippet"
    )
    assert error.error_type == ParseErrorType.INVALID_XML
    assert error.message == "Test message"
    assert error.context == "context snippet"
    assert "invalid_xml" in str(error)


def test_parse_error_str_representation():
    """Test ParseError string representation includes context."""
    error = ParseError(
        ParseErrorType.INVALID_JSON,
        "Malformed JSON",
        "long context string that should be truncated to 100 chars"
    )
    error_str = str(error)
    assert "[invalid_json]" in error_str
    assert "Malformed JSON" in error_str
    assert "context:" in error_str


def test_parse_error_empty_context():
    """Test ParseError string representation without context."""
    error = ParseError(
        ParseErrorType.INVALID_JSON,
        "Malformed JSON"
    )
    error_str = str(error)
    assert "[invalid_json]" in error_str
    assert "Malformed JSON" in error_str
    assert "context:" not in error_str


def test_xml_parse_error():
    """Test XMLParseError instantiation."""
    error = XMLParseError("XML parsing failed", "<broken>xml")
    assert error.error_type == ParseErrorType.INVALID_XML
    assert error.message == "XML parsing failed"
    assert error.context == "<broken>xml"
    assert "invalid_xml" in str(error)


def test_json_parse_error():
    """Test JSONParseError instantiation."""
    error = JSONParseError("JSON parsing failed", '{"broken": json}')
    assert error.error_type == ParseErrorType.INVALID_JSON
    assert error.message == "JSON parsing failed"
    assert error.context == '{"broken": json}'
    assert "invalid_json" in str(error)


def test_validation_error_default_type():
    """Test ValidationError with default error type."""
    error = ValidationError("Tool name missing", "<tool></tool>")
    assert error.error_type == ParseErrorType.MISSING_TOOL_NAME
    assert error.message == "Tool name missing"
    assert error.context == "<tool></tool>"
    assert "missing_tool_name" in str(error)


def test_validation_error_custom_type():
    """Test ValidationError with custom error type."""
    error = ValidationError(
        "Invalid arguments",
        '{"args": invalid}',
        error_type=ParseErrorType.INVALID_ARGUMENTS
    )
    assert error.error_type == ParseErrorType.INVALID_ARGUMENTS
    assert error.message == "Invalid arguments"
    assert error.context == '{"args": invalid}'
    assert "invalid_arguments" in str(error)

