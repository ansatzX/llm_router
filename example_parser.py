import json
import os
import inspect
import re
from openai import OpenAI
from json_repair import repair_json

def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get weather information for a specified location (simulated)
    
    Args:
        location: Location name
        unit: Temperature unit, either celsius or fahrenheit
    
    Returns:
        JSON string with weather information
    """
    weather_data = {
        "London": {"temperature": 15, "condition": "sunny", "humidity": 45},
        "New York": {"temperature": 20, "condition": "cloudy", "humidity": 60},
        "Tokyo": {"temperature": 25, "condition": "rainy", "humidity": 75},
    }
    weather = weather_data.get(location, {"temperature": 18, "condition": "unknown", "humidity": 50})
    if unit == "fahrenheit":
        weather["temperature"] = weather["temperature"] * 9/5 + 32
        weather["unit"] = "°F"
    else:
        weather["unit"] = "°C"
    return json.dumps(weather, ensure_ascii=False)

def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression
    
    Args:
        expression: Mathematical expression, e.g., "2 + 3 * 4"
    
    Returns:
        Calculation result
    """
    try:
        result = eval(expression)
        return json.dumps({"result": result, "expression": expression}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

tools = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "Location name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit, default is celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "calculate", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Mathematical expression to calculate, e.g., '2 + 3 * 4'"}}, "required": ["expression"]}}}
]

available_functions = {"get_weather": get_weather, "calculate": calculate}

def parse_mcp_tool_call(response_text: str):
    """Parse MCP-style tool call from model response. Returns first tool call or None."""
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
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Failed to parse arguments JSON: {e}, attempting to repair...")
            try:
                repaired = repair_json(args_match.group(1).strip())
                arguments = json.loads(repaired)
                print(f"✅  Successfully repaired JSON")
            except Exception as repair_error:
                print(f"❌  Failed to repair JSON: {repair_error}")
                arguments = {}
    else:
        arguments = {}
    if server_name and tool_name:
        return {"server_name": server_name, "tool_name": tool_name, "arguments": arguments}
    return None

def generate_mcp_system_prompt(openai_tools: list, available_functions: dict = None, server_name: str = "default", date: str = "2025-11-27") -> str:
    """Generate MCP-style system prompt from OpenAI tools format."""
    prefix = f"""You are MiroThinker, an advanced AI assistant developed by MiroMind.

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

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:

## Server name: {server_name}
"""
    tools_section = []
    for i, tool in enumerate(openai_tools):
        if tool.get("type") == "function":
            func = tool["function"]
            tool_name = func["name"]
            func_obj = available_functions[tool_name]
            full_description = inspect.getdoc(func_obj) or func.get("description", "")
            if i > 0:
                tools_section.append("\n")
            tools_section.append(f"### Tool name: {tool_name}\nDescription: {full_description}\n\nInput JSON schema: {json.dumps(func['parameters'], ensure_ascii=False)}\n")
    suffix = "\n# General Objective\n\nYou accomplish a given task iteratively, breaking it down into clear steps and working through them methodically."
    return prefix + ''.join(tools_section) + suffix

def run_conversation(user_query: str, model: str = "MiroThinker"):
    """Run a complete conversation with tool calling"""
    system_prompt = generate_mcp_system_prompt(openai_tools=tools, available_functions=available_functions, server_name="My-Tools", date="2025-12-01")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"), base_url=os.environ.get("BASE_URL", "your-base-url-here"))
    print(f"\n{'='*60}\nUser Query: {user_query}\n{'='*60}\n")
    messages = [{'role': 'system', 'content': system_prompt}, {"role": "user", "content": user_query}]
    print("📤 Sending request to model...")
    response = client.chat.completions.create(model=model, messages=messages)
    response_message = response.choices[0].message
    response_content = response_message.content
    tool_call = parse_mcp_tool_call(response_content)
    print(f"📝 Model response:\n{response_content}\n")
    messages.append(response_message)
    if tool_call:
        server_name = tool_call["server_name"]
        tool_name = tool_call["tool_name"]
        function_args = tool_call["arguments"]
        print(f"\n🔧 Model decided to call tool:\n  - Server: {server_name}\n    Tool: {tool_name}\n    Args: {json.dumps(function_args, ensure_ascii=False)}")
        function_response = available_functions[tool_name](**function_args)
        print(f"    Result: {function_response}\n")
        messages.append({"role": "user", "content": function_response})
        print("📤 Requesting model to generate final response based on tool results...\n")
        second_response = client.chat.completions.create(model=model, messages=messages)
        final_message = second_response.choices[0].message.content
        print(f"💬 Final Response:\n{final_message}\n")
        return final_message
    else:
        print(f"💬 Model Response (no tool calls):\n{response_message.content}\n")
        return response_message.content

def main():
    """Run multiple examples"""
    run_conversation("What's the weather like in London?")
    # run_conversation("Calculate (25 + 15) * 3 - 10")

if __name__ == "__main__":
    main()
