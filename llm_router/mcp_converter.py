"""
MCP XML format conversion utilities.

This module handles conversion between:
- OpenAI tools format -> MCP system prompt (for backend LLM)
- MCP XML tool calls (<use_mcp_tool>) -> OpenAI/Anthropic tool_use format

The backend LLM (e.g., MiroThinker on SGLang) uses MCP XML format for tool calls.
This module enables standard OpenAI/Anthropic API clients to work with such backends.
"""

import json
import re
import uuid
from json_repair import repair_json


def generate_mcp_system_prompt(tools: list, server_name: str = "tools") -> str:
    """
    Generate MCP-style system prompt from OpenAI tools format.

    This prompt instructs the LLM to use <use_mcp_tool> XML format for tool calls,
    which can then be parsed and converted back to standard API formats.

    Args:
        tools: List of tools in OpenAI format [{"type": "function", "function": {...}}]
        server_name: Server name to use in MCP format (default: "tools")

    Returns:
        System prompt string with MCP tool instructions, or empty string if no tools
    """
    if not tools:
        return ""

    prefix = """In this environment you have access to a set of tools you can use to answer the user's question.

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use.

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{
  "param1": "value1",
  "param2": "value2"
}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed at the end of your response, top-level, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

Here are the functions available:

"""

    tools_section = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            tool_name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
            tools_section.append(
                f"## Server: {server_name}\n"
                f"### Tool: {tool_name}\n"
                f"Description: {description}\n"
                f"Input schema: {json.dumps(parameters, ensure_ascii=False)}\n"
            )

    return prefix + "\n".join(tools_section)


def strip_think_tags(response_text: str) -> tuple[str, str]:
    """
    Extract and remove <think> tags from response text.

    Some models (like MiroThinker) wrap reasoning in <think> tags.
    This function removes them from the final output.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Tuple of (cleaned_text, thinking_content)
    """
    thinking_content = ""
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if think_match:
        thinking_content = think_match.group(1).strip()

    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    return cleaned.strip(), thinking_content


def is_anthropic_format(request_json: dict) -> bool:
    """
    Detect if request is in Anthropic format vs OpenAI format.

    Key differences:
    - Anthropic: {"type": "image"} blocks with source.data
    - OpenAI: {"type": "image_url"} blocks with image_url.url

    Args:
        request_json: The incoming request JSON

    Returns:
        True if Anthropic format, False if OpenAI format
    """
    messages = request_json.get("messages", [])
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                block_type = block.get("type", "")
                if block_type == "image":
                    return True
                elif block_type == "image_url":
                    return False

    # Check for Anthropic-style tools (name + input_schema instead of type: function)
    if "max_tokens" in request_json and "model" in request_json:
        if "tools" in request_json:
            tools = request_json["tools"]
            if tools and "input_schema" in tools[0]:
                return True

    return False


def _parse_arguments(args_text: str) -> dict:
    """
    Parse JSON arguments with fallback to json_repair for malformed JSON.

    Args:
        args_text: Raw arguments string from MCP XML

    Returns:
        Parsed arguments dict, or empty dict on failure
    """
    try:
        return json.loads(args_text.strip())
    except json.JSONDecodeError:
        try:
            repaired = repair_json(args_text.strip())
            return json.loads(repaired)
        except Exception:
            return {}


def extract_tool_calls_from_content(response_text: str) -> list:
    """
    Extract all MCP tool calls from response text.

    Parses <use_mcp_tool> XML blocks and extracts tool information.

    Args:
        response_text: Raw text response containing MCP XML

    Returns:
        List of dicts with keys: server_name, tool_name, arguments
    """
    # Limit input size to prevent ReDoS attacks
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if len(response_text) > MAX_SIZE:
        response_text = response_text[:MAX_SIZE]

    tool_calls = []
    pattern = r'<use_mcp_tool>(.*?)</use_mcp_tool>'
    matches = re.findall(pattern, response_text, re.DOTALL)

    for content in matches:
        server_match = re.search(r'<server_name>(.*?)</server_name>', content, re.DOTALL)
        tool_match = re.search(r'<tool_name>(.*?)</tool_name>', content, re.DOTALL)
        args_match = re.search(r'<arguments>(.*?)</arguments>', content, re.DOTALL)

        server_name = server_match.group(1).strip() if server_match else None
        tool_name = tool_match.group(1).strip() if tool_match else None
        arguments = _parse_arguments(args_match.group(1)) if args_match else {}

        if server_name and tool_name:
            tool_calls.append({
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            })

    return tool_calls


def strip_mcp_tags(response_text: str) -> str:
    """
    Remove MCP XML tags from response text.

    Args:
        response_text: Raw text with MCP XML tags

    Returns:
        Cleaned text with <use_mcp_tool> blocks removed
    """
    cleaned = re.sub(r'<use_mcp_tool>.*?</use_mcp_tool>', '', response_text, flags=re.DOTALL)
    return cleaned.strip()


def mcp_to_openai_tool_calls(response_text: str) -> list:
    """
    Convert MCP XML tool calls to OpenAI format.

    OpenAI format:
        message.tool_calls = [{
            "id": "call_xxx",
            "type": "function",
            "function": {"name": "...", "arguments": "..."}
        }]

    Args:
        response_text: Raw text containing MCP XML

    Returns:
        List of OpenAI-format tool_calls
    """
    mcp_calls = extract_tool_calls_from_content(response_text)
    openai_calls = []

    for call in mcp_calls:
        openai_calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": call["tool_name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False)
            }
        })

    return openai_calls


def mcp_to_anthropic_tool_use_blocks(response_text: str) -> list:
    """
    Convert MCP XML tool calls to Anthropic content blocks.

    Anthropic format:
        content = [{
            "type": "tool_use",
            "id": "toolu_xxx",
            "name": "...",
            "input": {...}
        }]

    Args:
        response_text: Raw text containing MCP XML

    Returns:
        List of Anthropic tool_use content blocks
    """
    mcp_calls = extract_tool_calls_from_content(response_text)
    blocks = []

    for call in mcp_calls:
        blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:8]}",
            "name": call["tool_name"],
            "input": call["arguments"]
        })

    return blocks


def build_anthropic_content_blocks(response_text: str) -> list:
    """
    Build complete Anthropic content blocks from response text.

    Creates content array with text (MCP tags removed) and tool_use blocks.

    Args:
        response_text: Raw text containing MCP XML

    Returns:
        List of Anthropic content blocks (text and/or tool_use)
    """
    blocks = []
    tool_blocks = mcp_to_anthropic_tool_use_blocks(response_text)
    text_content = strip_mcp_tags(response_text)

    if text_content:
        blocks.append({"type": "text", "text": text_content})

    blocks.extend(tool_blocks)
    return blocks


def convert_anthropic_messages_to_openai(messages: list) -> list:
    """
    Convert Anthropic message format to OpenAI format.

    Handles content arrays, image blocks, tool_use blocks, etc.

    Args:
        messages: List of messages in Anthropic format

    Returns:
        List of messages in OpenAI format
    """
    openai_messages = []

    for msg in messages:
        openai_msg = {"role": msg["role"]}

        if isinstance(msg.get("content"), str):
            openai_msg["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            content_blocks = msg["content"]
            text_parts = []
            tool_calls = []

            for block in content_blocks:
                block_type = block.get("type", "")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))

                elif block_type == "image":
                    # Convert Anthropic image to OpenAI format
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        image_url = f"data:{media_type};base64,{data}"

                        if "content" not in openai_msg:
                            openai_msg["content"] = []
                        if not isinstance(openai_msg["content"], list):
                            current = openai_msg["content"]
                            openai_msg["content"] = [{"type": "text", "text": current}] if current else []

                        openai_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })

                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })

                elif block_type == "tool_result":
                    if isinstance(block.get("content"), str):
                        text_parts.append(f"Tool result: {block.get('content')}")

            # Set content
            if isinstance(openai_msg.get("content"), list):
                if text_parts:
                    text_content = "\n".join(text_parts)
                    openai_msg["content"].insert(0, {"type": "text", "text": text_content})
            elif text_parts:
                openai_msg["content"] = "\n".join(text_parts)
            else:
                openai_msg["content"] = ""

            if tool_calls and msg["role"] == "assistant":
                openai_msg["tool_calls"] = tool_calls

        openai_messages.append(openai_msg)

    return openai_messages
