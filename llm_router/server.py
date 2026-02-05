"""
Flask server for the LLM Router.

This module provides API endpoints compatible with both OpenAI and Anthropic protocols.
It routes requests to a backend LLM (e.g., SGLang) that uses MCP XML format for tool calls.

Key features:
- Converts OpenAI/Anthropic tools to MCP system prompt
- Parses MCP XML responses back to standard API formats
- Strips <think> reasoning tags from model output
- Validates content based on model capabilities (text vs multimodal)
"""

import json
import os
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from urllib.request import Request, urlopen

from .mcp_converter import (
    mcp_to_openai_tool_calls,
    build_anthropic_content_blocks,
    convert_anthropic_messages_to_openai,
    generate_mcp_system_prompt,
    generate_lazy_mcp_prompt,
    get_tool_definitions_by_names,
    extract_tool_calls_from_content,
    is_anthropic_format,
)
from .llm_client import make_llm_request
from .model_config import (
    get_model_type,
    get_text_model_media_prompt,
    get_multimodal_document_prompt,
    validate_content_blocks,
)

logger = logging.getLogger(__name__)

# Default configuration from environment
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
LLM_API_KEY = os.environ.get("LLM_API_KEY", None)
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5001"))
# Max tokens cap - prevents requests exceeding model context length
# Set to empty string or 0 to disable cap
_max_tokens_cap_str = os.environ.get("MAX_TOKENS_CAP", "16384")
MAX_TOKENS_CAP = int(_max_tokens_cap_str) if _max_tokens_cap_str else None
# Max tools definition size in characters (roughly 4 chars per token)
# If tools exceed this, use lazy loading instead of full MCP prompt
MAX_TOOLS_CHARS = int(os.environ.get("MAX_TOOLS_CHARS", "40000"))
# Max rounds for search_tools internal loop
MAX_TOOL_SEARCH_ROUNDS = int(os.environ.get("MAX_TOOL_SEARCH_ROUNDS", "20"))
# Max message content chars (roughly 4 chars per token)
# Default 800k chars ≈ 200k tokens (Claude's context window)
# For smaller models, set MAX_CONTEXT_CHARS in .env (e.g., 52000 for 21k token model)
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "800000"))

# Debug mode for detailed logging
DEBUG_MODE = False
DEBUG_LOG_FILE = "llm_router.log"


def set_debug_mode(enabled: bool):
    """Enable or disable debug logging to file."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


def log_debug(message: str, data: dict = None):
    """Log debug message to file if debug mode is enabled."""
    if not DEBUG_MODE:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] {message}\n")
        if data:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.write("\n")


def truncate_messages(messages: list, max_chars: int = None) -> list:
    """
    Truncate messages to fit within context limit.

    Strategy:
    - Always keep the first message (usually system prompt)
    - Always keep the last message (current user request)
    - Remove older messages from the middle until under limit
    - If a single message is too long, truncate its content

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_chars: Maximum total characters (defaults to MAX_CONTEXT_CHARS)

    Returns:
        Truncated list of messages
    """
    if max_chars is None:
        max_chars = MAX_CONTEXT_CHARS

    def get_content_len(msg):
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        elif isinstance(content, list):
            return sum(len(str(c.get("text", ""))) for c in content if isinstance(c, dict))
        return 0

    def get_total_chars(msgs):
        return sum(get_content_len(m) for m in msgs)

    total = get_total_chars(messages)
    if total <= max_chars:
        return messages

    print(f"[Truncate] Messages exceed limit: {total} > {max_chars}, truncating...")

    # If only 1-2 messages, can't remove middle - truncate content
    if len(messages) <= 2:
        truncated = []
        remaining = max_chars
        for i, msg in enumerate(messages):
            msg_copy = msg.copy()
            content = msg_copy.get("content", "")
            if isinstance(content, str) and len(content) > remaining:
                # Keep last portion (more relevant for recent context)
                msg_copy["content"] = "...[truncated]..." + content[-(remaining - 20):]
                remaining = 0
            else:
                remaining -= get_content_len(msg_copy)
            truncated.append(msg_copy)
        return truncated

    # Keep first (system) and last (current request), remove from middle
    result = [messages[0]]  # First message
    middle = messages[1:-1]  # Middle messages
    last = messages[-1]  # Last message

    # Calculate space needed for first and last
    first_len = get_content_len(messages[0])
    last_len = get_content_len(last)
    available = max_chars - first_len - last_len

    if available <= 0:
        # First + last alone exceed limit, truncate last message
        print(f"[Truncate] First+last exceed limit, truncating last message")
        last_copy = last.copy()
        content = last_copy.get("content", "")
        if isinstance(content, str):
            max_last = max_chars - first_len - 100  # Leave some margin
            if max_last > 0:
                last_copy["content"] = content[:max_last] + "...[truncated]"
            else:
                last_copy["content"] = content[:1000] + "...[truncated]"
        return [messages[0], last_copy]

    # Add middle messages from most recent, until limit
    kept_middle = []
    used = 0
    for msg in reversed(middle):
        msg_len = get_content_len(msg)
        if used + msg_len <= available:
            kept_middle.insert(0, msg)
            used += msg_len
        else:
            # Can't fit this message - try truncating it
            remaining = available - used
            if remaining > 500:  # Worth keeping a truncated version
                msg_copy = msg.copy()
                content = msg_copy.get("content", "")
                if isinstance(content, str):
                    msg_copy["content"] = content[:remaining - 20] + "...[truncated]"
                    kept_middle.insert(0, msg_copy)
            break

    result.extend(kept_middle)
    result.append(last)

    new_total = get_total_chars(result)
    removed = len(messages) - len(result)
    print(f"[Truncate] Removed {removed} messages, {total} -> {new_total} chars")

    return result


app = Flask(__name__)


def create_app(llm_base_url=None, llm_api_key=None):
    """Create and configure the Flask application."""
    global LLM_BASE_URL, LLM_API_KEY
    if llm_base_url is not None:
        LLM_BASE_URL = llm_base_url
    if llm_api_key is not None:
        LLM_API_KEY = llm_api_key
    return app


def inject_system_prompt(messages: list, prompt: str) -> list:
    """Inject a system prompt at the beginning of messages."""
    if not prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = prompt + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": prompt})
    return messages


def log_request(endpoint: str, data: dict) -> tuple[list, int]:
    """Log request and return tools info."""
    tools = data.get("tools", [])
    messages_list = data.get("messages", [])
    total_chars = sum(len(str(m.get("content", ""))) for m in messages_list)
    tools_chars = len(json.dumps(tools)) if tools else 0

    print(f"[{endpoint}] messages={len(messages_list)}, tools={len(tools)}, content_chars={total_chars}, tools_chars={tools_chars}")
    log_debug(f"REQUEST /v1/{endpoint.lower()}", {
        "model": data.get("model"),
        "max_tokens": data.get("max_tokens"),
        "temperature": data.get("temperature"),
        "messages": messages_list,
        "tools_count": len(tools),
        "tools": tools,
    })
    return tools, tools_chars


def create_error_response(message: str, is_anthropic: bool = False, status_code: int = 500):
    """Create error response in appropriate format."""
    logger.error(f"Server error: {message}")

    if is_anthropic:
        return jsonify({
            "type": "error",
            "error": {
                "type": "server_error",
                "message": "An internal error occurred. Please try again later."
            }
        }), status_code
    else:
        return jsonify({
            "error": {
                "type": "server_error",
                "message": "An internal error occurred. Please try again later."
            }
        }), status_code


def process_messages_with_content_validation(messages, is_anthropic: bool = False):
    """
    Validate message content based on model capabilities.

    Returns:
        Tuple of (processed_messages, rejection_response)
    """
    for msg in messages:
        content = msg.get("content")
        if not content or isinstance(content, str):
            continue

        if isinstance(content, list):
            is_valid, response_data = validate_content_blocks(content, is_anthropic)
            if not is_valid:
                return None, build_rejection_response(response_data, is_anthropic)

    return messages, None


def build_rejection_response(rejection_data, is_anthropic: bool):
    """Build response when content cannot be processed by the model."""
    message = rejection_data.get("message", "Content cannot be processed by this model.")

    if is_anthropic:
        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": "llm-router",
            "content": [{"type": "text", "text": message}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": len(message.split())}
        }
    else:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": int(uuid.uuid1().time),
            "model": "llm-router",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": message},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(message.split()),
                "total_tokens": len(message.split())
            }
        }


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI Chat Completions API endpoint."""
    try:
        data = request.get_json()
        tools, tools_chars = log_request("OpenAI", data)

        model = data.get("model", "default")
        messages = data.get("messages", [])

        # Cap max_tokens if limit is configured
        max_tokens = data.get("max_tokens")
        if MAX_TOKENS_CAP is not None and max_tokens is not None and max_tokens > MAX_TOKENS_CAP:
            data["max_tokens"] = MAX_TOKENS_CAP

        # Validate content
        messages, rejection = process_messages_with_content_validation(messages, is_anthropic=False)
        if rejection:
            return jsonify(rejection)

        # Build request payload - start with original data, replace converted fields
        payload = {k: v for k, v in data.items() if k not in ("messages", "tools")}
        payload["model"] = model
        payload["messages"] = messages

        # Inject MCP system prompt if tools provided and not too large
        if tools and tools_chars < MAX_TOOLS_CHARS:
            mcp_prompt = generate_mcp_system_prompt(tools)
            payload["messages"] = inject_system_prompt(messages, mcp_prompt)
        elif tools and tools_chars >= MAX_TOOLS_CHARS:
            print(f"[OpenAI] Skipping MCP prompt: tools_chars={tools_chars} >= MAX_TOOLS_CHARS={MAX_TOOLS_CHARS}")

        # Forward to backend
        llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)

        # Process response
        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        # Always try to parse tool calls from response (regardless of tools param)
        tool_calls = mcp_to_openai_tool_calls(response_text)

        response = {
            "id": llm_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            "object": "chat.completion",
            "created": llm_response.get("created", 0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "tool_calls" if tool_calls else choice.get("finish_reason", "stop")
            }],
            "usage": llm_response.get("usage", {})
        }

        if tool_calls:
            response["choices"][0]["message"]["tool_calls"] = tool_calls

        # Log response
        log_debug("RESPONSE /v1/chat/completions", response)

        return jsonify(response)

    except Exception as e:
        logger.exception("Error in chat_completions")
        return create_error_response(str(e), is_anthropic=False)


@app.route('/v1/messages', methods=['POST'])
def messages():
    """Anthropic Messages API endpoint with lazy tool loading."""
    try:
        data = request.get_json()
        tools, tools_chars = log_request("Anthropic", data)

        model = data.get("model", "default")
        messages = data.get("messages", [])
        tools = data.get("tools", [])

        # Cap max_tokens if limit is configured
        max_tokens = data.get("max_tokens")
        if MAX_TOKENS_CAP is not None and max_tokens is not None and max_tokens > MAX_TOKENS_CAP:
            data["max_tokens"] = MAX_TOKENS_CAP

        # Validate content
        messages, rejection = process_messages_with_content_validation(messages, is_anthropic=True)
        if rejection:
            return jsonify(rejection)

        # Convert to OpenAI format for backend
        openai_messages = convert_anthropic_messages_to_openai(messages)

        # Truncate messages if they exceed context limit
        original_count = len(openai_messages)
        openai_messages = truncate_messages(openai_messages)
        if len(openai_messages) < original_count:
            log_debug("TRUNCATED messages", {
                "original_count": original_count,
                "new_count": len(openai_messages),
                "new_chars": sum(len(str(m.get("content", ""))) for m in openai_messages)
            })

        # Convert Anthropic tools to OpenAI format
        openai_tools = []
        if tools:
            for tool in tools:
                if tool.get("name") and tool.get("input_schema"):
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool["input_schema"]
                        }
                    })

        # Build request payload - start with original data, replace converted fields
        payload = {k: v for k, v in data.items() if k not in ("messages", "tools", "system")}
        payload["model"] = model
        payload["messages"] = openai_messages

        # Decide: full MCP prompt or lazy loading
        use_lazy_loading = openai_tools and tools_chars >= MAX_TOOLS_CHARS

        if openai_tools and not use_lazy_loading:
            # Normal mode: inject full MCP prompt
            mcp_prompt = generate_mcp_system_prompt(openai_tools)
            payload["messages"] = inject_system_prompt(openai_messages, mcp_prompt)

            # Forward to backend
            llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)
            response_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        elif use_lazy_loading:
            # Lazy loading mode: internal loop for search_tools
            print(f"[Anthropic] Using lazy loading: tools_chars={tools_chars}")

            # Inject lazy MCP prompt (tool names only + search_tools)
            lazy_prompt = generate_lazy_mcp_prompt(openai_tools)
            payload["messages"] = inject_system_prompt(openai_messages, lazy_prompt)

            # Internal loop
            response_text = ""
            for round_num in range(MAX_TOOL_SEARCH_ROUNDS):
                print(f"[Anthropic] Lazy loading round {round_num + 1}/{MAX_TOOL_SEARCH_ROUNDS}")

                llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)
                response_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Check for tool calls
                tool_calls = extract_tool_calls_from_content(response_text)
                real_tool_calls = [c for c in tool_calls if c["tool_name"] != "search_tools"]
                search_calls = [c for c in tool_calls if c["tool_name"] == "search_tools"]

                # If there are real tool calls (not search_tools), exit loop and return to client
                if real_tool_calls:
                    print(f"[Anthropic] Found real tool call(s): {[c['tool_name'] for c in real_tool_calls]}, exiting lazy loading at round {round_num + 1}")
                    break

                if not search_calls:
                    # No tool calls at all - exit loop
                    print(f"[Anthropic] Lazy loading complete at round {round_num + 1} (no tool calls)")
                    break

                # Process search_tools call
                search_call = search_calls[0]
                requested_names = search_call["arguments"].get("tool_names", [])
                print(f"[Anthropic] search_tools requested: {requested_names}")

                # Get tool definitions
                tool_defs = get_tool_definitions_by_names(openai_tools, requested_names)

                # Append assistant message and tool result to conversation
                payload["messages"].append({
                    "role": "assistant",
                    "content": response_text
                })
                payload["messages"].append({
                    "role": "user",
                    "content": f"""Here are the tool definitions you requested:

{tool_defs}

You now have the full definitions. Call the tool directly using <use_mcp_tool> format. Do NOT call search_tools again for these tools."""
                })

                log_debug(f"SEARCH_TOOLS round {round_num + 1}", {
                    "requested": requested_names,
                    "definitions": tool_defs[:500] + "..." if len(tool_defs) > 500 else tool_defs
                })
            else:
                # Max rounds exceeded
                print(f"[Anthropic] Max rounds exceeded ({MAX_TOOL_SEARCH_ROUNDS})")
                payload["messages"].append({
                    "role": "user",
                    "content": "No suitable tool found after maximum search attempts. Please respond without using tools."
                })
                llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)
                response_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        else:
            # No tools - just forward
            llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)
            response_text = llm_response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Build response
        if tools:
            content_blocks = build_anthropic_content_blocks(response_text)
            tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
            # Filter out search_tools from response (internal use only)
            tool_blocks = [b for b in tool_blocks if b.get("name") != "search_tools"]
            content_blocks = [b for b in content_blocks if b.get("type") != "tool_use" or b.get("name") != "search_tools"]

            stop_reason = "tool_use" if tool_blocks else "end_turn"
            response = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": model,
                "stop_reason": stop_reason,
                "usage": llm_response.get("usage", {"input_tokens": 0, "output_tokens": 0})
            }
        else:
            response = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
                "model": model,
                "stop_reason": "end_turn",
                "usage": llm_response.get("usage", {"input_tokens": 0, "output_tokens": 0})
            }

        # Log response
        log_debug("RESPONSE /v1/messages", response)

        return jsonify(response)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] messages endpoint: {e}")
        print(error_trace)
        log_debug("ERROR /v1/messages", {"error": str(e), "traceback": error_trace})
        logger.exception("Error in messages endpoint")
        return create_error_response(str(e), is_anthropic=True)


@app.route('/v1/chat', methods=['POST'])
def unified_chat():
    """Unified endpoint that auto-detects protocol format."""
    try:
        data = request.get_json()
        if is_anthropic_format(data):
            return messages()
        else:
            return chat_completions()
    except Exception as e:
        logger.exception("Error in unified_chat")
        return create_error_response(str(e))


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models from backend."""
    try:
        url = f"{LLM_BASE_URL}/v1/models"
        headers = {'Content-Type': 'application/json'}
        if LLM_API_KEY:
            headers['Authorization'] = f'Bearer {LLM_API_KEY}'
        req = Request(url, headers=headers)

        with urlopen(req) as response:
            return response.read()
    except Exception:
        return jsonify({
            "object": "list",
            "data": [{"id": "default", "object": "model", "created": 0}]
        })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "llm_base_url": LLM_BASE_URL,
        "model_type": get_model_type(),
        "max_tokens_cap": MAX_TOKENS_CAP
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information."""
    return jsonify({
        "name": "LLM Router",
        "description": "API router with OpenAI and Anthropic protocol support",
        "endpoints": {
            "unified": "/v1/chat",
            "openai": "/v1/chat/completions",
            "anthropic": "/v1/messages",
            "models": "/v1/models",
            "health": "/health"
        }
    })


# Alias routes to support various base_url configurations
# When client uses base_url like http://host/v1/chat, it appends /v1/messages
@app.route('/v1/chat/v1/messages', methods=['POST'])
def messages_via_chat():
    """Alias: /v1/chat as base + /v1/messages"""
    return messages()


@app.route('/v1/chat/v1/chat/completions', methods=['POST'])
def completions_via_chat():
    """Alias: /v1/chat as base + /v1/chat/completions"""
    return chat_completions()


@app.route('/v1/chat/v1/models', methods=['GET'])
def models_via_chat():
    """Alias: /v1/chat as base + /v1/models"""
    return list_models()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
