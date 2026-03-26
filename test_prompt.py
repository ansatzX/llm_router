#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test to verify MCP system prompt alignment."""

from llm_router.mcp_converter import generate_mcp_system_prompt

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location name"}
                },
                "required": ["location"]
            }
        }
    }
]

prompt = generate_mcp_system_prompt(tools, "My-Tools")

# Check key improvements
checks = [
    ("Today's date", "Today is:" in prompt),
    ("MCP description", "Model Context Protocol (MCP)" in prompt),
    ("Server name header", "## Server name: My-Tools" in prompt),
    ("Tool name format", "### Tool name:" in prompt),
    ("JSON schema format", "Input JSON schema:" in prompt),
    ("General objective", "General Objective" in prompt),
    ("Escaped quotes example", '\\"escaped string\\"' in prompt),
]

print("MCP System Prompt Alignment Checks:")
print("=" * 60)
for name, passed in checks:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name}")

if all(p for _, p in checks):
    print("\n[SUCCESS] All checks passed! Prompt is aligned with example.")
else:
    print("\n[ERROR] Some checks failed!")

print(f"\nPrompt length: {len(prompt)} characters")
print("\nFirst 800 characters:")
print("=" * 60)
print(prompt[:800])
print("=" * 60)
