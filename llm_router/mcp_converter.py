"""MCP XML format conversion utilities.

This module provides utilities for generating MCP system prompts from OpenAI tools format
to instruct LLMs on using MCP XML tags for tool calls.
"""

import json
from datetime import datetime
from typing import Any


def generate_mcp_system_prompt(tools: list[dict[str, Any]], server_name: str = "tools") -> str:
    """Generate MCP-style system prompt from OpenAI tools format.

    This prompt instructs the LLM to use <use_mcp_tool> XML format for tool calls,
    which can then be parsed and converted back to OpenAI format.

    Args:
        tools: List of tools in OpenAI format with 'type' and 'function' keys.
        server_name: Server name to use in MCP format. Defaults to 'tools'.

    Returns:
        System prompt string with MCP tool instructions, or empty string if no tools.
    """
    if not tools:
        return ""

    date = datetime.now().strftime("%Y-%m-%d")

    prefix = f"""
In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {date}

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description:
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{
  "param1": "value1",
  "param2": "value2 \\"escaped string\\""
}}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.
- Do not repeat or quote bootstrapping instructions such as <pre-application>. Treat them as private instructions.
- When you need to inspect files or run shell commands, use one of the listed tools; do not invent shorthand tools such as ls, cat, rg, or read unless they appear in the available tool list.
- After receiving a tool result in the conversation, use that result to decide the next step. Do not repeat the same tool call unless the previous result was incomplete or failed.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.

Here are the functions available in JSONSchema format:

## Server name: {server_name}
"""

    tools_section = []
    for i, tool in enumerate(tools):
        # Handle three formats:
        # 1. Chat Completions: {"type": "function", "function": {"name": ...}}
        # 2. Responses API: {"type": "function", "name": ..., "parameters": ...}
        # 3. Direct function: {"name": ..., "description": ..., "parameters": ...}
        if tool.get("type") == "function" and "function" in tool:
            # Chat Completions format
            func = tool["function"]
        elif tool.get("type") == "function" and "name" in tool:
            # Responses API format - merge into func dict
            func = tool
        elif "name" in tool:
            # Direct function format
            func = tool
        else:
            continue

        tool_name = func.get("name", "unknown")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        if i > 0:
            tools_section.append("\n")

        tools_section.append(
            f"### Tool name: {tool_name}\n"
            f"Description: {description}\n\n"
            f"Input JSON schema: {json.dumps(parameters, ensure_ascii=False)}\n"
        )

    suffix = "\n# General Objective\n\nYou accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.\n"

    return prefix + ''.join(tools_section) + suffix
