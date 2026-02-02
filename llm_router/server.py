"""
Flask server module for the LLM Router.

This module contains the Flask application and route handlers for the
OpenAI and Anthropic protocol endpoints.
"""

import json
import os
import uuid
from flask import Flask, request, jsonify
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .mcp_converter import (
    mcp_to_openai_tool_calls,
    mcp_to_anthropic_tool_use_blocks,
    build_anthropic_content_blocks,
    convert_anthropic_messages_to_openai,
)
from .llm_client import make_llm_request

# Create Flask app
app = Flask(__name__)

# Configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
LLM_API_KEY = os.environ.get("LLM_API_KEY", None)
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5001"))


def create_app(llm_base_url=None, llm_api_key=None):
    """Create and configure the Flask application.

    Args:
        llm_base_url: Optional LLM base URL. If not provided, uses env var or default.
        llm_api_key: Optional LLM API key. If not provided, uses env var or None.

    Returns:
        The configured Flask app.
    """
    global LLM_BASE_URL, LLM_API_KEY
    if llm_base_url:
        LLM_BASE_URL = llm_base_url
    if llm_api_key:
        LLM_API_KEY = llm_api_key
    return app


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI Chat Completions API endpoint."""
    try:
        data_in = request.get_json()

        # Extract request parameters
        model = data_in.get("model", "default")
        messages = data_in.get("messages", [])
        tools = data_in.get("tools", [])
        temperature = data_in.get("temperature", 0.7)
        max_tokens = data_in.get("max_tokens", None)

        # Build LLM request payload
        llm_payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            llm_payload["tools"] = tools

        if max_tokens:
            llm_payload["max_tokens"] = max_tokens

        # Forward request to LLM backend
        llm_response = make_llm_request(llm_payload, LLM_BASE_URL, LLM_API_KEY)

        # Extract response content
        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        # Check for MCP tool calls in response
        tool_calls = mcp_to_openai_tool_calls(response_text)
        text_blocks = [b for b in build_anthropic_content_blocks(response_text) if b.get("type") == "text"]
        text_content = text_blocks[0].get("text", "") if text_blocks else ""

        # Build OpenAI-style response
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

        # Add tool_calls if present
        if tool_calls:
            response["choices"][0]["message"]["tool_calls"] = tool_calls

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }), 500


@app.route('/v1/messages', methods=['POST'])
def messages():
    """Anthropic Messages API endpoint."""
    # Check for authentication (any value is accepted for LLM backends without key)
    api_key = request.headers.get('x-api-key') or request.headers.get('Authorization')

    if api_key:
        # Extract token if using Bearer format
        if api_key.lower().startswith('bearer '):
            api_key = api_key[7:]

    try:
        data_in = request.get_json()

        # Extract request parameters
        model = data_in.get("model", "default")
        messages = data_in.get("messages", [])
        tools = data_in.get("tools", [])
        temperature = data_in.get("temperature", 0.7)
        max_tokens = data_in.get("max_tokens", 4096)

        # Convert Anthropic messages to OpenAI format for LLM backend
        openai_messages = convert_anthropic_messages_to_openai(messages)

        # Convert Anthropic tools format to OpenAI format if present
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

        # Build LLM request payload
        llm_payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if openai_tools:
            llm_payload["tools"] = openai_tools

        # Forward request to LLM backend
        llm_response = make_llm_request(llm_payload, LLM_BASE_URL, LLM_API_KEY)

        # Extract response content
        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        # Build Anthropic-style content blocks
        content_blocks = build_anthropic_content_blocks(response_text)

        # Determine stop reason
        tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
        stop_reason = "tool_use" if tool_blocks else choice.get("finish_reason", "stop")

        # Build Anthropic-style response
        response = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "usage": llm_response.get("usage", {
                "input_tokens": 0,
                "output_tokens": 0
            })
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "type": "error",
            "error": {
                "type": "server_error",
                "message": str(e)
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models endpoint (compatible with both OpenAI and Anthropic)."""
    try:
        # Try to get models from LLM backend
        url = f"{LLM_BASE_URL}/v1/models"
        headers = {'Content-Type': 'application/json'}
        if LLM_API_KEY:
            headers['Authorization'] = f'Bearer {LLM_API_KEY}'
        req = Request(url, headers=headers)

        with urlopen(req) as response:
            return response.read()

    except Exception:
        # Return default model list if LLM backend is unavailable
        return jsonify({
            "object": "list",
            "data": [
                {
                    "id": "default",
                    "object": "model",
                    "created": 0,
                }
            ]
        })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "llm_base_url": LLM_BASE_URL
    })


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information."""
    return jsonify({
        "name": "LLM Router",
        "description": "Flask-based router with OpenAI and Anthropic protocol support",
        "endpoints": {
            "openai": "/v1/chat/completions",
            "anthropic": "/v1/messages",
            "models": "/v1/models",
            "health": "/health"
        },
        "documentation": "See README.md for details"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
