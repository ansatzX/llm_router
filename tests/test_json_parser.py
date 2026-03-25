"""Tests for JSON parser."""
from llm_router.parser.json_parser import JSONParser


def test_json_parser_format_name():
    """Test JSONParser format name."""
    parser = JSONParser()
    assert parser.format_name == "json"


def test_parse_valid_json_with_name():
    """Test parsing valid JSON with 'name' field."""
    parser = JSONParser()
    content = '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "get_weather"
    assert result.tool_calls[0].arguments == {"location": "Tokyo"}


def test_parse_valid_json_with_tool():
    """Test parsing valid JSON with 'tool' field."""
    parser = JSONParser()
    content = '{"tool": "read_file", "input": {"path": "/test.txt"}}'
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "/test.txt"}


def test_parse_multiple_json_objects():
    """Test parsing multiple JSON objects."""
    parser = JSONParser()
    content = '''
First: {"name": "tool1", "arguments": {"a": 1}}
Second: {"name": "tool2", "arguments": {"b": 2}}
'''
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].tool_name == "tool1"
    assert result.tool_calls[1].tool_name == "tool2"


def test_parse_malformed_json():
    """Test parsing malformed JSON uses json_repair."""
    parser = JSONParser()
    content = '{name: "test_tool", arguments: {param: "value"}}'
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].tool_name == "test_tool"


def test_parse_nested_json_string():
    """Test parsing JSON with nested JSON string in arguments."""
    parser = JSONParser()
    content = '{"name": "test", "arguments": "{\\"nested\\": \\"value\\"}"}'
    result = parser.parse(content)

    assert result.success is True
    assert result.tool_calls[0].arguments == {"nested": "value"}


def test_parse_missing_name_field():
    """Test parsing JSON without name/tool field fails."""
    parser = JSONParser()
    content = '{"some_field": "value"}'
    result = parser.parse(content)

    assert result.success is False
    assert len(result.tool_calls) == 0


def test_parse_non_object_json():
    """Test parsing non-object JSON fails."""
    parser = JSONParser()
    content = '["not", "an", "object"]'
    result = parser.parse(content)

    assert result.success is False


def test_parse_no_json_found():
    """Test parsing content with no JSON returns empty result."""
    parser = JSONParser()
    result = parser.parse("This is just plain text")

    assert result.success is False
    assert len(result.tool_calls) == 0


def test_parse_ignores_non_tool_json():
    """Test that JSON without tool-related fields is ignored."""
    parser = JSONParser()
    content = '''
{"config": {"setting": "value"}}
{"name": "real_tool", "arguments": {"x": 1}}
{"metadata": "ignored"}
'''
    result = parser.parse(content)

    assert result.success is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "real_tool"
