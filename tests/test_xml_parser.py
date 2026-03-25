"""Tests for XML parser."""
from llm_router.parser.xml_parser import XMLParser


def test_xml_parser_format_name():
    """Test XMLParser format name."""
    parser = XMLParser()
    assert parser.format_name == "xml"


def test_parse_valid_mcp_xml():
    """Test parsing valid MCP XML format."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>
{"location": "London"}
</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert tool_call.tool_name == "get_weather"
    assert tool_call.arguments == {"location": "London"}
    assert tool_call.server_name == "tools"


def test_parse_mcp_xml_missing_server_name():
    """Test MCP XML with missing server_name uses default."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>test_tool</tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].server_name == "tools"


def test_parse_tool_call_xml_format():
    """Test parsing TOOL_CALL XML format."""
    parser = XMLParser()
    content = """[TOOL_CALL]
{"name": "read_file", "arguments": {"path": "/test.txt"}}
[/TOOL_CALL]"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "read_file"


def test_parse_simple_tool_xml():
    """Test parsing simple <tool> XML format."""
    parser = XMLParser()
    content = """
<tool>
<function_name>calculate</function_name>
<parameters>{"expr": "2+2"}</parameters>
</tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "calculate"
    assert result.tool_calls[0].arguments == {"expr": "2+2"}


def test_parse_multiple_tool_calls():
    """Test parsing multiple tool calls in same format."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>tool1</tool_name>
<arguments>{"a": 1}</arguments>
</use_mcp_tool>
<use_mcp_tool>
<tool_name>tool2</tool_name>
<arguments>{"b": 2}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "tool1"
    assert result.tool_calls[1].tool_name == "tool2"


def test_parse_malformed_json_arguments():
    """Test parsing with malformed JSON uses json_repair."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>test</tool_name>
<arguments>{location: "London", unit: "celsius"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].arguments == {"location": "London", "unit": "celsius"}


def test_parse_missing_tool_name():
    """Test parsing XML with missing tool_name (format fallback to JSON)."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    # XMLParser should fail (no valid tool calls)
    # But parse_tool_calls() will try JSON parser next
    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_parse_no_xml_found():
    """Test parsing content with no XML returns empty result."""
    parser = XMLParser()
    result = parser.parse("This is just plain text with no XML")

    assert result.success is False
    assert len(result.tool_calls) == 0
