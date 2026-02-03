"""
MCP XML format conversion utilities.

This module provides functions for parsing and converting MCP XML format
tool calls to OpenAI and Anthropic protocol formats.

MCP XML Format (Input):
    <use_mcp_tool>
    <server_name>server name here</server_name>
    <tool_name>tool name here</tool_name>
    <arguments>{"key": "value"}</arguments>
    </use_mcp_tool>
"""

import json
import re
import uuid


def parse_mcp_tool_call(response_text: str) -> dict:
    """
    Parse a single MCP-style tool call from response text.

    Args:
        response_text: The raw text response containing MCP XML tags

    Returns:
        dict with keys: server_name, tool_name, arguments
        None if no valid tool call is found
    """
    match = re.search(r'<use_mcp_tool>(.*?)</use_mcp_tool>', response_text, re.DOTALL)
    if not match:
        return None

    content = match.group(1)
    server_match = re.search(r'<server_name>(.*?)</server_name>', content, re.DOTALL)
    tool_match = re.search(r'<tool_name>(.*?)</tool_name>', content, re.DOTALL)
    args_match = re.search(r'<arguments>(.*?)</arguments>', content, re.DOTALL)

    server_name = server_match.group(1).strip() if server_match else None
    tool_name = tool_match.group(1).strip() if tool_match else None

    if args_match:
        try:
            arguments = json.loads(args_match.group(1).strip())
        except json.JSONDecodeError:
            arguments = {}
    else:
        arguments = {}

    if server_name and tool_name:
        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments
        }
    return None


def extract_tool_calls_from_content(response_text: str) -> list:
    """
    Extract all MCP tool calls from the response text.

    Args:
        response_text: The raw text response containing MCP XML

    Returns:
        List of tool call dicts, each containing server_name, tool_name, arguments
    """
    tool_calls = []
    pattern = r'<use_mcp_tool>(.*?)</use_mcp_tool>'
    matches = re.findall(pattern, response_text, re.DOTALL)

    for content in matches:
        server_match = re.search(r'<server_name>(.*?)</server_name>', content, re.DOTALL)
        tool_match = re.search(r'<tool_name>(.*?)</tool_name>', content, re.DOTALL)
        args_match = re.search(r'<arguments>(.*?)</arguments>', content, re.DOTALL)

        server_name = server_match.group(1).strip() if server_match else None
        tool_name = tool_match.group(1).strip() if tool_match else None

        if args_match:
            try:
                arguments = json.loads(args_match.group(1).strip())
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = {}

        if server_name and tool_name:
            tool_calls.append({
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": arguments
            })

    return tool_calls


def strip_mcp_tags(response_text: str) -> str:
    """
    Remove MCP XML tags from response text, leaving only the natural language content.

    Args:
        response_text: The raw text response containing MCP XML

    Returns:
        Cleaned text with MCP tags removed
    """
    # Remove all MCP tool call blocks
    cleaned = re.sub(r'<use_mcp_tool>.*?</use_mcp_tool>', '', response_text, flags=re.DOTALL)
    return cleaned.strip()


def mcp_to_openai_tool_calls(response_text: str) -> list:
    """
    Convert MCP XML format tool calls to OpenAI format.

    OpenAI Format:
        message.tool_calls = [{
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": '{"key": "value"}'
            }
        }]

    Args:
        response_text: The raw text response containing MCP XML

    Returns:
        List of OpenAI tool_calls objects
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
    Convert MCP XML format tool calls to Anthropic content blocks.

    Anthropic Format:
        content = [{
            "type": "tool_use",
            "id": "toolu_xxx",
            "name": "tool_name",
            "input": {"key": "value"}
        }]

    Args:
        response_text: The raw text response containing MCP XML

    Returns:
        List of Anthropic tool_use content blocks
    """
    mcp_calls = extract_tool_calls_from_content(response_text)
    anthropic_blocks = []

    for call in mcp_calls:
        anthropic_blocks.append({
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:8]}",
            "name": call["tool_name"],
            "input": call["arguments"]
        })

    return anthropic_blocks


def build_anthropic_content_blocks(response_text: str) -> list:
    """
    Build complete Anthropic content blocks from response text.

    This function creates a content array that includes both the text content
    (with MCP tags removed) and any tool_use blocks.

    Args:
        response_text: The raw text response containing MCP XML

    Returns:
        List of Anthropic content blocks (text and/or tool_use)
    """
    blocks = []

    # Get tool use blocks
    tool_blocks = mcp_to_anthropic_tool_use_blocks(response_text)

    # Get cleaned text content
    text_content = strip_mcp_tags(response_text)

    # Add text block if there's any text content
    if text_content:
        blocks.append({"type": "text", "text": text_content})

    # Add tool use blocks
    blocks.extend(tool_blocks)

    return blocks


def convert_anthropic_messages_to_openai(messages: list) -> list:
    """
    Convert Anthropic message format to OpenAI message format.

    This handles the different message structures between the two protocols:
    - Anthropic: uses content array with text blocks and tool_result blocks
    - OpenAI: uses content string and tool_calls array

    Args:
        messages: List of messages in Anthropic format

    Returns:
        List of messages in OpenAI format
    """
    openai_messages = []

    for msg in messages:
        openai_msg = {"role": msg["role"]}

        # Handle content conversion
        if isinstance(msg.get("content"), str):
            # Simple string content
            openai_msg["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            # Content array - need to convert blocks
            content_blocks = msg["content"]
            text_parts = []
            tool_calls = []

            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    # Convert Anthropic image to OpenAI format
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        # Build data URL for OpenAI format
                        image_url = f"data:{media_type};base64,{data}"
                        # For OpenAI format, we need to keep the image in the content array
                        # Store it in a special format that the backend can understand
                        if "content" not in openai_msg:
                            openai_msg["content"] = []
                        if not isinstance(openai_msg["content"], list):
                            # Convert string to array
                            current_content = openai_msg["content"]
                            openai_msg["content"] = [{"type": "text", "text": current_content}] if current_content else []
                        openai_msg["content"].append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                elif block.get("type") == "tool_use":
                    # Convert tool_use to OpenAI tool_call format
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}))
                        }
                    })
                elif block.get("type") == "tool_result":
                    # Convert tool_result to assistant message with tool_call_id
                    # For now, we'll treat this as a user message with the result
                    if isinstance(block.get("content"), str):
                        text_parts.append(f"Tool result: {block.get('content')}")

            # Handle content setting
            # If content is already a list (from image processing), keep it
            # Otherwise, create content from text_parts
            if isinstance(openai_msg.get("content"), list):
                # Content is already a list (contains image_url or other blocks)
                # Add text parts as a text block if there are any
                if text_parts:
                    text_content = "\n".join(text_parts)
                    openai_msg["content"].insert(0, {"type": "text", "text": text_content})
            elif text_parts:
                # Only text parts, no list content
                openai_msg["content"] = "\n".join(text_parts)
            else:
                # No content at all
                openai_msg["content"] = ""

            # Add tool_calls if present
            if tool_calls and msg["role"] == "assistant":
                openai_msg["tool_calls"] = tool_calls

        openai_messages.append(openai_msg)

    return openai_messages
