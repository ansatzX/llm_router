"""Integration tests for parser public API."""
from llm_router.parser import ParseResult, ToolCall, parse_tool_calls


def test_parse_tool_calls_basic():
    """Test basic parse_tool_calls usage."""
    content = """
<use_mcp_tool>
<tool_name>test_tool</tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parse_tool_calls(content)

    assert isinstance(result, ParseResult)
    assert result.success is True
    assert len(result.tool_calls) == 1
    assert isinstance(result.tool_calls[0], ToolCall)


def test_parse_tool_calls_with_reasoning_content():
    """Test parsing with reasoning_content merged."""
    content = "Main response"
    reasoning = """
<use_mcp_tool>
<tool_name>hidden_tool</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
"""
    result = parse_tool_calls(content, reasoning)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "hidden_tool"


def test_parse_tool_calls_format_priority():
    """Test XML parser has priority over JSON."""
    content = """
<use_mcp_tool>
<tool_name>xml_tool</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
{"name": "json_tool", "arguments": {}}
"""
    result = parse_tool_calls(content)

    # Should find XML first and return it
    assert result.success is True
    assert result.tool_calls[0].tool_name == "xml_tool"


def test_parse_tool_calls_json_fallback():
    """Test JSON parser is used when XML fails validation."""
    content = '{"name": "json_only_tool", "arguments": {"x": 1}}'
    result = parse_tool_calls(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "json_only_tool"


def test_parse_tool_calls_format_fallback():
    """Test fallback to JSON when XML matches but validation fails."""
    content = """
<use_mcp_tool>
<arguments>{"x": 1}</arguments>
</use_mcp_tool>
{"name": "json_tool", "arguments": {"y": 2}}
"""
    result = parse_tool_calls(content)

    # Should fail XML (missing tool_name), succeed with JSON
    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "json_tool"


def test_parse_tool_calls_prefer_json():
    """Test prefer_format parameter."""
    content = """
{"name": "json_tool", "arguments": {}}
<use_mcp_tool>
<tool_name>xml_tool</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
"""
    result = parse_tool_calls(content, prefer_format="json")

    # Should try JSON first
    assert result.success is True
    assert result.tool_calls[0].tool_name == "json_tool"


def test_parse_tool_calls_no_tool_calls():
    """Test parsing content with no tool calls."""
    result = parse_tool_calls("Just regular text")

    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_parse_tool_calls_error_aggregation():
    """Test errors are aggregated from all parsers."""
    content = '{"invalid": "json"}'  # Missing name field

    result = parse_tool_calls(content)

    assert result.success is False
    assert len(result.errors) > 0


def test_parse_tool_calls_empty_content():
    """Test parsing empty content."""
    result = parse_tool_calls("")

    assert result.success is False


def test_exports():
    """Test that public API exports are correct."""
    from llm_router.parser import ParseError, ParseResult, ToolCall, parse_tool_calls

    assert callable(parse_tool_calls)
    assert ToolCall is not None
    assert ParseResult is not None
    assert ParseError is not None
