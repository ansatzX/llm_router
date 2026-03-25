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
