"""Tests for parser error types."""
import pytest
from llm_router.parser.errors import ParseErrorType, ParseError


def test_parse_error_type_enum():
    """Test ParseErrorType enum values."""
    assert ParseErrorType.EMPTY_INPUT.value == "empty_input"
    assert ParseErrorType.INVALID_XML.value == "invalid_xml"
    assert ParseErrorType.INVALID_JSON.value == "invalid_json"
    assert ParseErrorType.MISSING_TOOL_NAME.value == "missing_tool_name"
    assert ParseErrorType.INVALID_ARGUMENTS.value == "invalid_arguments"


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
