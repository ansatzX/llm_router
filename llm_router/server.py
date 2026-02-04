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
from urllib.error import URLError, HTTPError

from .mcp_converter import (
    mcp_to_openai_tool_calls,
    build_anthropic_content_blocks,
    convert_anthropic_messages_to_openai,
    generate_mcp_system_prompt,
    strip_think_tags,
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
MAX_TOKENS_CAP = int(os.environ.get("MAX_TOKENS_CAP", "4096"))
# Max tools definition size in characters (roughly 4 chars per token)
# If tools exceed this, skip MCP prompt to avoid context overflow
MAX_TOOLS_CHARS = int(os.environ.get("MAX_TOOLS_CHARS", "40000"))

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


app = Flask(__name__)


class AppConfig:
    """Application configuration container."""

    def __init__(self):
        self.llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
        self.llm_api_key = os.environ.get("LLM_API_KEY", None)
        self.flask_port = int(os.environ.get("FLASK_PORT", "5001"))
        self.max_tokens_cap = int(os.environ.get("MAX_TOKENS_CAP", "4096"))

    def update(self, llm_base_url=None, llm_api_key=None):
        if llm_base_url is not None:
            self.llm_base_url = llm_base_url
        if llm_api_key is not None:
            self.llm_api_key = llm_api_key


def create_app(llm_base_url=None, llm_api_key=None):
    """Create and configure the Flask application."""
    app.config_obj = AppConfig()
    app.config_obj.update(llm_base_url, llm_api_key)

    global LLM_BASE_URL, LLM_API_KEY
    LLM_BASE_URL = app.config_obj.llm_base_url
    LLM_API_KEY = app.config_obj.llm_api_key

    return app


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

        # Debug: log request
        tools = data.get("tools", [])
        tools_count = len(tools)
        messages_list = data.get("messages", [])
        messages_count = len(messages_list)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages_list)
        tools_chars = len(json.dumps(tools)) if tools else 0

        # Console output (brief)
        print(f"[OpenAI] messages={messages_count}, tools={tools_count}, content_chars={total_chars}, tools_chars={tools_chars}")

        # File logging (detailed)
        log_debug("REQUEST /v1/chat/completions", {
            "model": data.get("model"),
            "messages": messages_list,
            "tools_count": tools_count,
            "tools": tools,
        })

        model = data.get("model", "default")
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", None)

        # Cap max_tokens to prevent exceeding model context
        if max_tokens is None or max_tokens > MAX_TOKENS_CAP:
            max_tokens = MAX_TOKENS_CAP

        # Validate content
        messages, rejection = process_messages_with_content_validation(messages, is_anthropic=False)
        if rejection:
            return jsonify(rejection)

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Inject MCP system prompt if tools provided and not too large
        # Skip if tools would overflow model context
        if tools and tools_chars < MAX_TOOLS_CHARS:
            mcp_prompt = generate_mcp_system_prompt(tools)
            if mcp_prompt:
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = mcp_prompt + "\n\n" + messages[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": mcp_prompt})
            payload["messages"] = messages
        elif tools and tools_chars >= MAX_TOOLS_CHARS:
            print(f"[OpenAI] Skipping MCP prompt: tools_chars={tools_chars} >= MAX_TOOLS_CHARS={MAX_TOOLS_CHARS}")

        # Forward to backend
        llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)

        # Process response
        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        # Strip think tags
        response_text, _ = strip_think_tags(response_text)

        if tools:
            # Parse MCP tool calls
            tool_calls = mcp_to_openai_tool_calls(response_text)
            from .mcp_converter import strip_mcp_tags
            text_content = strip_mcp_tags(response_text)

            response = {
                "id": llm_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                "object": "chat.completion",
                "created": llm_response.get("created", 0),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text_content,
                    },
                    "finish_reason": "tool_calls" if tool_calls else choice.get("finish_reason", "stop")
                }],
                "usage": llm_response.get("usage", {})
            }

            if tool_calls:
                response["choices"][0]["message"]["tool_calls"] = tool_calls
        else:
            # No tools - return cleaned response
            response = {
                "id": llm_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                "object": "chat.completion",
                "created": llm_response.get("created", 0),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": choice.get("finish_reason", "stop")
                }],
                "usage": llm_response.get("usage", {})
            }

        # Log response
        log_debug("RESPONSE /v1/chat/completions", response)

        return jsonify(response)

    except Exception as e:
        logger.exception("Error in chat_completions")
        return create_error_response(str(e), is_anthropic=False)


@app.route('/v1/messages', methods=['POST'])
def messages():
    """Anthropic Messages API endpoint."""
    try:
        data = request.get_json()

        # Debug: log request
        tools = data.get("tools", [])
        tools_count = len(tools)
        messages_list = data.get("messages", [])
        messages_count = len(messages_list)
        total_chars = sum(len(str(m.get("content", ""))) for m in messages_list)
        tools_chars = len(json.dumps(tools)) if tools else 0

        # Console output (brief)
        print(f"[Anthropic] messages={messages_count}, tools={tools_count}, content_chars={total_chars}, tools_chars={tools_chars}")

        # File logging (detailed)
        log_debug("REQUEST /v1/messages", {
            "model": data.get("model"),
            "messages": messages_list,
            "tools_count": tools_count,
            "tools": tools,
        })

        model = data.get("model", "default")
        messages = data.get("messages", [])
        tools = data.get("tools", [])
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 4096)

        # Cap max_tokens to prevent exceeding model context
        if max_tokens > MAX_TOKENS_CAP:
            max_tokens = MAX_TOKENS_CAP

        # Validate content
        messages, rejection = process_messages_with_content_validation(messages, is_anthropic=True)
        if rejection:
            return jsonify(rejection)

        # Convert to OpenAI format for backend
        openai_messages = convert_anthropic_messages_to_openai(messages)

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

        # Build request payload
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Inject MCP system prompt if tools provided and not too large
        # Skip if tools would overflow model context
        if openai_tools and tools_chars < MAX_TOOLS_CHARS:
            mcp_prompt = generate_mcp_system_prompt(openai_tools)
            if mcp_prompt:
                if openai_messages and openai_messages[0].get("role") == "system":
                    openai_messages[0]["content"] = mcp_prompt + "\n\n" + openai_messages[0]["content"]
                else:
                    openai_messages.insert(0, {"role": "system", "content": mcp_prompt})
            payload["messages"] = openai_messages
        elif tools_chars >= MAX_TOOLS_CHARS:
            print(f"[Anthropic] Skipping MCP prompt: tools_chars={tools_chars} >= MAX_TOOLS_CHARS={MAX_TOOLS_CHARS}")

        # Forward to backend
        llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)

        # Process response
        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        # Strip think tags
        response_text, _ = strip_think_tags(response_text)

        if tools:
            # Build Anthropic content blocks
            content_blocks = build_anthropic_content_blocks(response_text)
            tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
            stop_reason = "tool_use" if tool_blocks else choice.get("finish_reason", "stop")

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
            # No tools - return cleaned response
            response = {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
                "model": model,
                "stop_reason": choice.get("finish_reason", "stop"),
                "usage": llm_response.get("usage", {"input_tokens": 0, "output_tokens": 0})
            }

        # Log response
        log_debug("RESPONSE /v1/messages", response)

        return jsonify(response)

    except Exception as e:
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
