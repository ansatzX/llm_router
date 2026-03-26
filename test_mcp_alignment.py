#!/usr/bin/env python3
"""Test script to verify MCP parsing alignment with the example code."""

import json
from llm_router.mcp_converter import generate_mcp_system_prompt, extract_tool_calls_from_content

# Test tools (same as example)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to calculate"}
                },
                "required": ["expression"]
            }
        }
    }
]

# Test 1: Generate MCP system prompt
print("=" * 60)
print("Test 1: MCP System Prompt Generation")
print("=" * 60)
prompt = generate_mcp_system_prompt(tools, server_name="My-Tools")
print(prompt[:500] + "...\n[truncated for brevity]\n")

# Test 2: Parse MCP XML tool call (same format as example)
print("=" * 60)
print("Test 2: Parse MCP XML Tool Call")
print("=" * 60)
mcp_response = """
<use_mcp_tool>
<server_name>My-Tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>
{"location": "London"}
</arguments>
</use_mcp_tool>
"""
tool_calls = extract_tool_calls_from_content(mcp_response)
print(f"Found {len(tool_calls)} tool call(s):")
for call in tool_calls:
    print(f"  - Server: {call['server_name']}")
    print(f"    Tool: {call['tool_name']}")
    print(f"    Args: {json.dumps(call['arguments'], ensure_ascii=False)}")
print()

# Test 3: Parse JSON format tool call
print("=" * 60)
print("Test 3: Parse JSON Format Tool Call")
print("=" * 60)
json_response = """{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "fahrenheit"}}"""
tool_calls = extract_tool_calls_from_content(json_response)
print(f"Found {len(tool_calls)} tool call(s):")
for call in tool_calls:
    print(f"  - Tool: {call['tool_name']}")
    print(f"    Args: {json.dumps(call['arguments'], ensure_ascii=False)}")
print()

# Test 4: Parse TOOL_CALL XML format
print("=" * 60)
print("Test 4: Parse TOOL_CALL XML Format")
print("=" * 60)
tool_call_response = """[TOOL_CALL]
{"name": "calculate", "arguments": {"expression": "(25 + 15) * 3"}}
[/TOOL_CALL]"""
tool_calls = extract_tool_calls_from_content(tool_call_response)
print(f"Found {len(tool_calls)} tool call(s):")
for call in tool_calls:
    print(f"  - Tool: {call['tool_name']}")
    print(f"    Args: {json.dumps(call['arguments'], ensure_ascii=False)}")
print()

# Test 5: Malformed JSON (should be repaired)
print("=" * 60)
print("Test 5: Malformed JSON Repair")
print("=" * 60)
malformed_response = """
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>
{location: "New York", unit: "celsius"}
</arguments>
</use_mcp_tool>
"""
tool_calls = extract_tool_calls_from_content(malformed_response)
print(f"Found {len(tool_calls)} tool call(s):")
for call in tool_calls:
    print(f"  - Tool: {call['tool_name']}")
    print(f"    Args: {json.dumps(call['arguments'], ensure_ascii=False)}")
print()

print("=" * 60)
print("All tests passed!")
print("=" * 60)
