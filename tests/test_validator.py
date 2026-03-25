"""Tests for tool call validator."""
import pytest
from llm_router.parser.base import ToolCall
from llm_router.parser.validator import validate_tool_call, sanitize_arguments


def test_validate_valid_tool_call():
    """Test validation of valid tool call."""
    tool_call = ToolCall(
        tool_name="test_tool",
        arguments={"param": "value"},
        server_name="test_server"
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is True
    assert warning is None


def test_validate_empty_tool_name():
    """Test validation fails for empty tool name."""
    tool_call = ToolCall(
        tool_name="",
        arguments={}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "empty" in warning.lower()


def test_validate_non_string_tool_name():
    """Test validation fails for non-string tool name."""
    tool_call = ToolCall(
        tool_name=123,  # type: ignore
        arguments={}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "string" in warning.lower()


def test_validate_non_dict_arguments():
    """Test validation fails for non-dict arguments."""
    tool_call = ToolCall(
        tool_name="test",
        arguments="not a dict"  # type: ignore
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "dict" in warning.lower()


def test_validate_whitespace_tool_name():
    """Test validation fails for whitespace-only tool name."""
    tool_call = ToolCall(
        tool_name="   ",
        arguments={}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is False
    assert "whitespace" in warning.lower()


def test_validate_none_values_in_arguments():
    """Test warning for None values in arguments."""
    tool_call = ToolCall(
        tool_name="test",
        arguments={"param1": "value", "param2": None}
    )
    is_valid, warning = validate_tool_call(tool_call)

    assert is_valid is True
    assert warning is not None
    assert "None" in warning


def test_sanitize_arguments_removes_none():
    """Test sanitize_arguments removes None values."""
    args = {"a": 1, "b": None, "c": "value", "d": None}
    sanitized = sanitize_arguments(args)

    assert sanitized == {"a": 1, "c": "value"}


def test_sanitize_arguments_empty_dict():
    """Test sanitize_arguments with empty dict."""
    assert sanitize_arguments({}) == {}


def test_sanitize_arguments_non_dict():
    """Test sanitize_arguments returns empty dict for non-dict."""
    assert sanitize_arguments("not a dict") == {}
    assert sanitize_arguments(None) == {}
    assert sanitize_arguments([1, 2, 3]) == {}
