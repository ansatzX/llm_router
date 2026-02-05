"""
MCP XML format conversion utilities.

This module handles conversion between:
- OpenAI tools format -> MCP system prompt (for backend LLM)
- Tool call XML/JSON -> OpenAI/Anthropic tool_use format

Supported tool call formats:
- MCP XML: <use_mcp_tool><tool_name>...</tool_name><arguments>...</arguments></use_mcp_tool>
- Tool call XML: <tool_call><tool_name>...</tool_name><arguments>...</arguments></tool_call>
- Simple XML: <tool><function_name>...</function_name><arguments>...</arguments></tool>
- JSON: {"name": "...", "arguments": {...}}

The backend LLM may use any of these formats for tool calls.
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
    Extract all tool calls from response text.

    Supports multiple XML formats and JSON format:
    - MCP XML: <use_mcp_tool>...</use_mcp_tool>
    - Tool call XML: <tool_call>...</tool_call>
    - Simple XML: <tool>...</tool>
    - JSON: {"name": "tool", "arguments": {...}}

    Args:
        response_text: Raw text response containing tool calls

    Returns:
        List of dicts with keys: server_name, tool_name, arguments
    """
    # Limit input size to prevent ReDoS attacks
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if len(response_text) > MAX_SIZE:
        response_text = response_text[:MAX_SIZE]

    tool_calls = []

    # XML patterns to try in order: (pattern, has_server_name)
    XML_PATTERNS = [
        (r'<use_mcp_tool>(.*?)</use_mcp_tool>', True),   # MCP format
        (r'<tool_call>(.*?)</tool_call>', False),        # tool_call format
        (r'<tool>(.*?)</tool>', False),                  # simple tool format
    ]

    for pattern, has_server in XML_PATTERNS:
        matches = re.findall(pattern, response_text, re.DOTALL)
        for content in matches:
            if has_server:
                server_match = re.search(r'<server_name>(.*?)</server_name>', content, re.DOTALL)
                server_name = server_match.group(1).strip() if server_match else "tools"
                tool_match = re.search(r'<tool_name>(.*?)</tool_name>', content, re.DOTALL)
                tool_name = tool_match.group(1).strip() if tool_match else None
                args_match = re.search(r'<arguments>(.*?)</arguments>', content, re.DOTALL)
                arguments = _parse_arguments(args_match.group(1)) if args_match else {}
            else:
                server_name = "tools"
                tool_name = _extract_tool_name_from_xml(content)
                arguments = _extract_arguments_from_xml(content)

            if tool_name:
                tool_calls.append({
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "arguments": arguments
                })
        if tool_calls:
            break

    # Try generic XML format: <tag_name>...<arguments>...</arguments></tag_name>
    if not tool_calls:
        pattern = r'<([a-zA-Z][a-zA-Z0-9_-]*)>(.*?)</\1>'
        matches = re.findall(pattern, response_text, re.DOTALL)

        for tag_name, content in matches:
            if tag_name in ['think', 'thinking', 'response', 'output', 'result']:
                continue
            arguments = _extract_arguments_from_xml(content)
            if arguments:
                tool_calls.append({
                    "server_name": "tools",
                    "tool_name": tag_name,
                    "arguments": arguments
                })

    # If no XML found, try JSON format
    if not tool_calls:
        tool_calls.extend(_extract_json_tool_calls(response_text))

    return tool_calls


def _extract_tool_name_from_xml(content: str) -> str | None:
    """Extract tool name from XML content, trying multiple tag names."""
    # Try various tool name tags
    for tag in ['tool_name', 'function_name', 'name', 'function', 'action']:
        match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _extract_arguments_from_xml(content: str) -> dict:
    """Extract arguments from XML content, trying multiple tag names."""
    # Try various argument tags
    for tag in ['arguments', 'parameters', 'input', 'params', 'args']:
        match = re.search(rf'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
        if match:
            return _parse_arguments(match.group(1))
    return {}


def _extract_json_tool_calls(response_text: str) -> list:
    """
    Extract JSON format tool calls from response text.

    Handles formats like:
    - {"name": "Read", "arguments": {...}}
    - {"tool": "Read", "input": {...}}

    Args:
        response_text: Raw text response

    Returns:
        List of tool call dicts
    """
    tool_calls = []

    # Find all potential JSON objects by matching balanced braces
    def find_json_objects(text):
        objects = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Find matching closing brace
                depth = 1
                start = i
                i += 1
                while i < len(text) and depth > 0:
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objects.append(text[start:i])
            else:
                i += 1
        return objects

    json_candidates = find_json_objects(response_text)

    for candidate in json_candidates:
        # Check if it looks like a tool call
        if '"name"' not in candidate and '"tool"' not in candidate:
            continue

        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                repaired = repair_json(candidate)
                obj = json.loads(repaired)
            except Exception:
                continue

        # Must be a dict
        if not isinstance(obj, dict):
            continue

        # Extract tool name and arguments
        tool_name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
        arguments = obj.get("arguments") or obj.get("input") or obj.get("params") or {}

        if isinstance(arguments, str):
            arguments = _parse_arguments(arguments)

        if tool_name and isinstance(tool_name, str):
            tool_calls.append({
                "server_name": "tools",
                "tool_name": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {}
            })

    return tool_calls


def strip_mcp_tags(response_text: str) -> str:
    """
    Remove tool call content from response text.

    Handles MCP XML, tool_call XML, simple XML, and JSON format tool calls.

    Args:
        response_text: Raw text with tool calls

    Returns:
        Cleaned text with tool call blocks removed
    """
    # Remove MCP XML tags
    cleaned = re.sub(r'<use_mcp_tool>.*?</use_mcp_tool>', '', response_text, flags=re.DOTALL)

    # Remove <tool_call> XML tags
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)

    # Remove simple <tool> XML tags
    cleaned = re.sub(r'<tool>.*?</tool>', '', cleaned, flags=re.DOTALL)

    # Remove JSON tool calls (find balanced braces containing "name" or "tool")
    def remove_json_tool_calls(text):
        result = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                # Find matching closing brace
                depth = 1
                start = i
                i += 1
                while i < len(text) and depth > 0:
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    json_str = text[start:i]
                    # Check if it's a tool call JSON
                    if '"name"' in json_str or '"tool"' in json_str:
                        try:
                            obj = json.loads(json_str)
                            if isinstance(obj, dict) and (obj.get("name") or obj.get("tool")):
                                # Skip this JSON (don't add to result)
                                continue
                        except:
                            pass
                    result.append(json_str)
                else:
                    result.append(text[start:i])
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    cleaned = remove_json_tool_calls(cleaned)

    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
    return cleaned.strip()


def _mcp_to_tool_calls(response_text: str, format: str) -> list:
    """Internal: convert MCP XML to OpenAI or Anthropic format."""
    mcp_calls = extract_tool_calls_from_content(response_text)
    result = []
    for call in mcp_calls:
        if format == "openai":
            result.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": call["tool_name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False)
                }
            })
        else:  # anthropic
            result.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:8]}",
                "name": call["tool_name"],
                "input": call["arguments"]
            })
    return result


def mcp_to_openai_tool_calls(response_text: str) -> list:
    """Convert MCP XML tool calls to OpenAI format."""
    return _mcp_to_tool_calls(response_text, "openai")


def mcp_to_anthropic_tool_use_blocks(response_text: str) -> list:
    """Convert MCP XML tool calls to Anthropic content blocks."""
    return _mcp_to_tool_calls(response_text, "anthropic")


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


# search_tools 的完整定义
SEARCH_TOOLS_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_tools",
        "description": "获取工具的完整定义和参数。在使用任何工具前，必须先用此工具获取该工具的详细信息。可以一次查询多个工具，也可以多次调用直到找到合适的工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要查询的工具名称列表"
                }
            },
            "required": ["tool_names"]
        }
    }
}


def generate_lazy_mcp_prompt(tools: list, server_name: str = "tools") -> str:
    """
    Generate MCP prompt with only tool names + search_tools definition.

    This enables lazy loading of tool definitions - model sees all available
    tool names but must use search_tools to get full definitions before use.

    Args:
        tools: List of tools in OpenAI format
        server_name: Server name for MCP format

    Returns:
        System prompt with tool names list and search_tools definition
    """
    if not tools:
        return ""

    # Extract tool names
    tool_names = []
    for tool in tools:
        if tool.get("type") == "function":
            name = tool["function"].get("name", "unknown")
            tool_names.append(name)

    prompt = """In this environment you have access to a set of tools you can use to answer the user's question.

# Tool-Use Formatting Instructions

Tool-use is formatted using XML-style tags:

<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{"param1": "value1", "param2": "value2"}
</arguments>
</use_mcp_tool>

Important: Tool-use must be placed at the end of your response.

# Available Tools (names only)

The following tools are available. You MUST use search_tools to get a tool's full definition before using it:

"""

    # Add tool names as a list
    for name in tool_names:
        prompt += f"- {name}\n"

    # Add search_tools full definition
    search_func = SEARCH_TOOLS_DEFINITION["function"]
    prompt += f"""
# Tool Discovery

To use any tool above, first get its definition:

## Tool: search_tools
Description: {search_func["description"]}
Input schema: {json.dumps(search_func["parameters"], ensure_ascii=False)}

Example - get definitions for Read and Write tools:
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>search_tools</tool_name>
<arguments>
{{"tool_names": ["Read", "Write"]}}
</arguments>
</use_mcp_tool>
"""

    return prompt


def get_tool_definitions_by_names(tools: list, names: list, server_name: str = "tools") -> str:
    """
    Get full definitions for specified tools by name.

    Args:
        tools: List of all tools in OpenAI format
        names: List of tool names to get definitions for
        server_name: Server name for formatting

    Returns:
        Formatted string with full tool definitions
    """
    result_parts = []
    names_set = set(names)
    found_names = set()

    # Special handling for search_tools (virtual tool, not in original tools list)
    if "search_tools" in names_set:
        found_names.add("search_tools")
        search_func = SEARCH_TOOLS_DEFINITION["function"]
        result_parts.append(
            f"## Tool: search_tools\n"
            f"Description: {search_func['description']}\n"
            f"Input schema: {json.dumps(search_func['parameters'], ensure_ascii=False)}\n"
        )

    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            tool_name = func.get("name", "")

            if tool_name in names_set:
                found_names.add(tool_name)
                description = func.get("description", "")
                parameters = func.get("parameters", {})

                result_parts.append(
                    f"## Tool: {tool_name}\n"
                    f"Description: {description}\n"
                    f"Input schema: {json.dumps(parameters, ensure_ascii=False)}\n"
                )

    # Report not found tools
    not_found = names_set - found_names
    if not_found:
        result_parts.append(f"\n[Not found: {', '.join(not_found)}]")

    if not result_parts:
        return "No matching tools found."

    return "\n".join(result_parts)
