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


def test_parse_codex_tool_calls_xml_format():
    """Test parsing Codex-style XML tool_calls blocks."""
    parser = XMLParser()
    content = """
<tool_calls>
  <tool_call name="read">
    <parameter name="filePath" string="true">/Users/test/file.txt</parameter>
  </tool_call>
</tool_calls>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "read"
    assert result.tool_calls[0].arguments == {"filePath": "/Users/test/file.txt"}


def test_parse_bare_codex_tool_call_blocks():
    """Test parsing multiple bare Codex-style tool_call blocks."""
    parser = XMLParser()
    content = """
<tool_call name="ls">
  <parameter name="path" string="true">/Users/ansatz/data/code/MOKIT</parameter>
</tool_call>
<tool_call name="rg">
  <parameter name="pattern" string="true">MOKIT</parameter>
  <parameter name="path" string="true">/Users/ansatz/data/code/MOKIT/README.md</parameter>
  <parameter name="max_count" string="false">10</parameter>
</tool_call>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "ls"
    assert result.tool_calls[0].arguments == {
        "path": "/Users/ansatz/data/code/MOKIT",
    }
    assert result.tool_calls[1].tool_name == "rg"
    assert result.tool_calls[1].arguments == {
        "pattern": "MOKIT",
        "path": "/Users/ansatz/data/code/MOKIT/README.md",
        "max_count": 10,
    }


def test_codex_tool_call_parameter_types():
    """Test string=false parameters are parsed as JSON scalars."""
    parser = XMLParser()
    content = """
<tool_call name="example">
  <parameter name="limit" string="false">10</parameter>
  <parameter name="recursive" string="false">true</parameter>
  <parameter name="label" string="true">10</parameter>
</tool_call>
"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].arguments == {
        "limit": 10,
        "recursive": True,
        "label": "10",
    }


def test_unknown_codex_tool_call_fails_when_tools_are_known():
    """Unknown model-invented tool names should trigger retry instead of execution."""
    parser = XMLParser()
    content = """
<tool_call name="ls">
  <parameter name="path" string="true">/tmp</parameter>
</tool_call>
"""
    result = parser.parse(
        content,
        available_tools=[
            {
                "type": "function",
                "function": {"name": "exec_command", "parameters": {"type": "object"}},
            }
        ],
    )

    assert result.success is False
    assert result.tool_calls == []
    assert "Unknown tool 'ls'" in result.errors[0]
    assert "exec_command" in result.errors[0]


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


def test_parse_tool_call_json_missing_name_field():
    """Test TOOL_CALL JSON with missing 'name' field."""
    parser = XMLParser()
    content = """[TOOL_CALL]
{"arguments": {"path": "/test.txt"}}
[/TOOL_CALL]"""
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_parse_tool_call_json_non_dict_arguments():
    """Test TOOL_CALL with non-dict arguments."""
    parser = XMLParser()
    content = """[TOOL_CALL]
{"name": "read_file", "arguments": "not a dict"}
[/TOOL_CALL]"""
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].arguments == {}  # Non-dict replaced with empty dict


def test_parse_tool_call_completely_unparseable_json():
    """Test TOOL_CALL with completely unparseable JSON."""
    parser = XMLParser()
    content = """[TOOL_CALL]
this is not json at all!!! $$$
[/TOOL_CALL]"""
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_parse_xml_unparseable_arguments_after_repair():
    """Test XML with unparseable arguments even after json_repair."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>test_tool</tool_name>
<arguments>!!! completely broken json $$$</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0
    assert len(result.errors) > 0


def test_size_limit_enforced():
    """Test that 10MB size limit is enforced."""
    parser = XMLParser()
    # Create content larger than 10MB
    large_content = "x" * (11 * 1024 * 1024)  # 11MB
    result = parser.parse(large_content)

    # Should not crash, just truncate
    assert result.success is False


def test_empty_tool_name_string():
    """Test XML with empty tool_name string."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name></tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0


def test_tool_name_whitespace_only():
    """Test XML with tool_name containing only whitespace."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>   </tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0


def test_mixed_valid_and_invalid_tool_calls():
    """Test parsing mixed valid and invalid tool calls in same content."""
    parser = XMLParser()
    content = """
<use_mcp_tool>
<tool_name>valid_tool</tool_name>
<arguments>{"param": "value"}</arguments>
</use_mcp_tool>
<use_mcp_tool>
<tool_name></tool_name>
<arguments>{"param": "value2"}</arguments>
</use_mcp_tool>
<use_mcp_tool>
<tool_name>another_valid</tool_name>
<arguments>{"x": "y"}</arguments>
</use_mcp_tool>
"""
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 2  # Only valid ones
    assert result.tool_calls[0].tool_name == "valid_tool"
    assert result.tool_calls[1].tool_name == "another_valid"


def test_empty_arguments_vs_missing_arguments_tag():
    """Test difference between empty arguments dict and missing arguments tag."""
    parser = XMLParser()

    # Empty arguments dict
    content1 = """
<use_mcp_tool>
<tool_name>test</tool_name>
<arguments>{}</arguments>
</use_mcp_tool>
"""
    result1 = parser.parse(content1)
    assert result1.success is True
    assert result1.tool_calls[0].arguments == {}

    # Missing arguments tag
    content2 = """
<use_mcp_tool>
<tool_name>test</tool_name>
</use_mcp_tool>
"""
    result2 = parser.parse(content2)
    assert result2.success is True
    assert result2.tool_calls[0].arguments == {}
