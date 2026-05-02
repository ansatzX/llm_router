"""Focused tests for parser error formatting and specialization."""

from llm_router.parser.errors import ParseError, ParseErrorType, ValidationError


def test_parse_error_str_includes_type_message_and_context_preview():
    error = ParseError(
        ParseErrorType.INVALID_JSON,
        "Malformed JSON",
        "x" * 150,
    )

    error_str = str(error)

    assert "[invalid_json]" in error_str
    assert "Malformed JSON" in error_str
    assert "context:" in error_str
    assert len(error_str) < 180


def test_validation_error_defaults_and_custom_types():
    default_error = ValidationError("Tool name missing", "<tool></tool>")
    custom_error = ValidationError(
        "Invalid arguments",
        '{"args": invalid}',
        error_type=ParseErrorType.INVALID_ARGUMENTS,
    )

    assert default_error.error_type == ParseErrorType.MISSING_TOOL_NAME
    assert custom_error.error_type == ParseErrorType.INVALID_ARGUMENTS
