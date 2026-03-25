"""Tests for parser base data classes."""
from llm_router.parser.base import ParseResult, ToolCall


def test_tool_call_creation():
    """Test ToolCall dataclass creation."""
    tool_call = ToolCall(
        tool_name="get_weather",
        arguments={"location": "London"},
        server_name="tools"
    )
    assert tool_call.tool_name == "get_weather"
    assert tool_call.arguments == {"location": "London"}
    assert tool_call.server_name == "tools"


def test_tool_call_default_server_name():
    """Test ToolCall default server_name."""
    tool_call = ToolCall(
        tool_name="test_tool",
        arguments={}
    )
    assert tool_call.server_name == "tools"


def test_tool_call_to_openai_format():
    """Test ToolCall conversion to OpenAI format."""
    tool_call = ToolCall(
        tool_name="read_file",
        arguments={"path": "/test/file.txt"}
    )
    openai_format = tool_call.to_openai_format()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "read_file"
    assert openai_format["function"]["arguments"] == '{"path": "/test/file.txt"}'
    assert openai_format["id"].startswith("call_")


def test_parse_result_ok():
    """Test ParseResult.ok factory method."""
    tool_call = ToolCall(tool_name="test", arguments={})
    result = ParseResult.ok([tool_call])

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.errors == []
    assert result.warnings == []


def test_parse_result_ok_with_warnings():
    """Test ParseResult.ok with warnings."""
    tool_call = ToolCall(tool_name="test", arguments={})
    result = ParseResult.ok([tool_call], warnings=["Test warning"])

    assert result.success is True
    assert result.warnings == ["Test warning"]


def test_parse_result_error():
    """Test ParseResult.error factory method."""
    result = ParseResult.error(["Error 1", "Error 2"])

    assert result.success is False
    assert result.tool_calls == []
    assert result.errors == ["Error 1", "Error 2"]
    assert result.warnings == []


def test_parse_result_error_with_warnings():
    """Test ParseResult.error with warnings."""
    result = ParseResult.error(
        ["Error message"],
        warnings=["Warning message"]
    )

    assert result.success is False
    assert result.errors == ["Error message"]
    assert result.warnings == ["Warning message"]


def test_parse_result_ok_empty_tool_calls():
    """Test ParseResult.ok with empty tool_calls list."""
    result = ParseResult.ok([])

    assert result.success is True
    assert result.tool_calls == []
    assert result.errors == []
    assert result.warnings == []


def test_tool_call_special_characters():
    """Test ToolCall with special characters in name and arguments."""
    tool_call = ToolCall(
        tool_name="get_user_info",
        arguments={
            "name": "O'Brien",
            "email": "user@example.com",
            "message": "Hello\nWorld\t!",
            "unicode": "你好世界"
        }
    )

    assert tool_call.tool_name == "get_user_info"
    assert tool_call.arguments["name"] == "O'Brien"
    assert tool_call.arguments["message"] == "Hello\nWorld\t!"
    assert tool_call.arguments["unicode"] == "你好世界"

    # Test OpenAI format conversion preserves special characters
    openai_format = tool_call.to_openai_format()
    assert '"name": "O\'Brien"' in openai_format["function"]["arguments"]
    assert "你好世界" in openai_format["function"]["arguments"]


def test_tool_call_nested_arguments():
    """Test ToolCall with nested dictionary arguments."""
    tool_call = ToolCall(
        tool_name="search",
        arguments={
            "query": "test",
            "filters": {
                "date": "2024-01-01",
                "tags": ["python", "testing"]
            }
        }
    )

    openai_format = tool_call.to_openai_format()
    assert "filters" in openai_format["function"]["arguments"]


def test_parse_result_none_warnings():
    """Test ParseResult factory methods with explicit None warnings."""
    tool_call = ToolCall(tool_name="test", arguments={})

    # ok() with None warnings should default to empty list
    result_ok = ParseResult.ok([tool_call], warnings=None)
    assert result_ok.warnings == []

    # error() with None warnings should default to empty list
    result_error = ParseResult.error(["error"], warnings=None)
    assert result_error.warnings == []


def test_parse_result_immutability():
    """Test that ParseResult is frozen and cannot be modified."""
    tool_call = ToolCall(tool_name="test", arguments={})
    result = ParseResult.ok([tool_call])

    # Attempting to modify a frozen dataclass should raise an error
    try:
        result.success = False
        raise AssertionError("Should not be able to modify frozen dataclass")
    except AttributeError:
        pass  # Expected behavior

    try:
        result.tool_calls = []
        raise AssertionError("Should not be able to modify frozen dataclass")
    except AttributeError:
        pass  # Expected behavior


def test_parse_result_multiple_tool_calls():
    """Test ParseResult with multiple tool calls."""
    tool_calls = [
        ToolCall(tool_name="read_file", arguments={"path": "/test1.txt"}),
        ToolCall(tool_name="read_file", arguments={"path": "/test2.txt"}),
        ToolCall(tool_name="write_file", arguments={"path": "/test3.txt", "content": "data"})
    ]

    result = ParseResult.ok(tool_calls, warnings=["Processing 3 files"])

    assert result.success is True
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0].tool_name == "read_file"
    assert result.tool_calls[2].arguments["content"] == "data"
    assert result.warnings == ["Processing 3 files"]


def test_parse_result_multiple_errors():
    """Test ParseResult.error with multiple error messages."""
    errors = [
        "Failed to parse tool call at line 1",
        "Invalid JSON in arguments",
        "Missing required field: tool_name"
    ]

    result = ParseResult.error(errors)

    assert result.success is False
    assert len(result.errors) == 3
    assert "Invalid JSON" in result.errors[1]
    assert result.tool_calls == []
