"""Test schema validation for tool arguments."""

from llm_router.parser import parse_tool_calls


def test_schema_validation_missing_required_field():
    """Test that missing required fields are caught during parsing."""
    # Define tool schema with required field 'questions' (array)
    tools = [{
        "type": "function",
        "function": {
            "name": "AskUserQuestion",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": False
            }
        }
    }]

    # Model returns wrong format (question instead of questions)
    content = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>AskUserQuestion</tool_name>
<arguments>
{"question": "What would you like?"}
</arguments>
</use_mcp_tool>
"""

    result = parse_tool_calls(content, available_tools=tools)

    # Should fail validation
    assert not result.success
    assert any("Missing required parameter 'questions'" in err for err in result.errors)
    assert any("Unexpected parameter 'question'" in err for err in result.errors)


def test_schema_validation_valid_arguments():
    """Test that valid arguments pass schema validation."""
    tools = [{
        "type": "function",
        "function": {
            "name": "AskUserQuestion",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": False
            }
        }
    }]

    # Correct format
    content = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>AskUserQuestion</tool_name>
<arguments>
{"questions": [{"question": "What would you like?"}]}
</arguments>
</use_mcp_tool>
"""

    result = parse_tool_calls(content, available_tools=tools)

    # Should succeed
    assert result.success
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "AskUserQuestion"
    assert "questions" in result.tool_calls[0].arguments


def test_schema_validation_no_schema():
    """Test that validation is skipped when no schema is provided."""
    content = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>AskUserQuestion</tool_name>
<arguments>
{"question": "What would you like?"}
</arguments>
</use_mcp_tool>
"""

    # No tools schema provided
    result = parse_tool_calls(content, available_tools=None)

    # Should succeed (no schema to validate against)
    assert result.success
    assert len(result.tool_calls) == 1


def test_schema_validation_additional_properties():
    """Test that unexpected fields are caught when additionalProperties is false."""
    tools = [{
        "type": "function",
        "function": {
            "name": "simple_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"}
                },
                "required": ["required_field"],
                "additionalProperties": False
            }
        }
    }]

    content = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>simple_tool</tool_name>
<arguments>
{"required_field": "value", "unexpected_field": "value"}
</arguments>
</use_mcp_tool>
"""

    result = parse_tool_calls(content, available_tools=tools)

    # Should fail due to unexpected field
    assert not result.success
    assert any("Unexpected parameter 'unexpected_field'" in err for err in result.errors)


def test_schema_validation_json_format():
    """Test that JSON format also undergoes schema validation."""
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"}
                },
                "required": ["required_param"],
                "additionalProperties": False
            }
        }
    }]

    # JSON format with missing required field
    content = '{"name": "test_tool", "arguments": {"wrong_param": "value"}}'

    result = parse_tool_calls(content, available_tools=tools)

    # Should fail validation
    assert not result.success
    assert any("Missing required parameter 'required_param'" in err for err in result.errors)
