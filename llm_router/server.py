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
from .model_config import (
    get_model_type,
    is_multimodal,
    get_text_model_media_prompt,
    get_multimodal_document_prompt,
    validate_content_blocks,
    detect_content_type,
)

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

        # Validate messages for content that model cannot process
        validated_messages, rejection_response = process_messages_with_content_validation(
            messages, is_anthropic_format=False
        )

        if rejection_response:
            # Content was rejected, return the rejection response
            return jsonify(rejection_response)

        # Use validated messages (may have been modified)
        messages = validated_messages

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

        # Check for MCP tool calls in response only if tools were requested
        if tools:
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
        else:
            # If no tools were requested, return response as-is
            response = llm_response

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

        # Validate messages for content that model cannot process
        validated_messages, rejection_response = process_messages_with_content_validation(
            messages, is_anthropic_format=True
        )

        if rejection_response:
            # Content was rejected, return the rejection response
            return jsonify(rejection_response)

        # Use validated messages
        messages = validated_messages

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

        # Check for MCP tool calls in response only if tools were requested
        if tools:
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
        else:
            # If no tools were requested, return response as-is
            # Convert OpenAI format response to Anthropic format
            if "choices" in llm_response:
                choice = llm_response["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                response = {
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                    "model": model,
                    "stop_reason": choice.get("finish_reason", "stop"),
                    "usage": llm_response.get("usage", {
                        "input_tokens": 0,
                        "output_tokens": 0
                    })
                }
            else:
                response = llm_response

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
        "llm_base_url": LLM_BASE_URL,
        "model_type": get_model_type()
    })


def process_messages_with_content_validation(messages, is_anthropic_format=False):
    """
    Process messages and validate content based on model capabilities.

    Args:
        messages: List of messages
        is_anthropic_format: True if using Anthropic format

    Returns:
        Tuple of (processed_messages, rejection_response)
        - processed_messages: Messages if validation passed, None if rejected
        - rejection_response: Response dict if content rejected, None if accepted
    """
    model_type = get_model_type()

    for msg in messages:
        content = msg.get("content")
        if not content:
            continue

        # Handle string content (always text)
        if isinstance(content, str):
            continue

        # Handle list content (Anthropic format or complex content blocks)
        if isinstance(content, list):
            is_valid, response_data = validate_content_blocks(content, is_anthropic_format)
            if not is_valid:
                # Content was rejected
                return None, build_rejection_response(msg, response_data, is_anthropic_format)

    return messages, None


def build_rejection_response(original_msg, rejection_data, is_anthropic_format):
    """
    Build a rejection response when content cannot be processed.

    Args:
        original_msg: The message that was rejected
        rejection_data: Data about why it was rejected
        is_anthropic_format: True for Anthropic format, False for OpenAI

    Returns:
        Response dict in the appropriate format
    """
    rejection_type = rejection_data.get("type", "unknown")
    message = rejection_data.get("message", "Content cannot be processed by this model.")

    if is_anthropic_format:
        # Anthropic format response
        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": "llm-router",
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 0,
                "output_tokens": len(message.split())
            }
        }
    else:
        # OpenAI format response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": int(uuid.uuid1().time),
            "model": "llm-router",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(message.split()),
                "total_tokens": len(message.split())
            }
        }


@app.route('/v1/chat', methods=['POST'])
def unified_chat():
    """Unified chat endpoint that auto-detects protocol format."""
    from llm_router.mcp_converter import is_anthropic_format

    try:
        data_in = request.get_json()

        # Auto-detect protocol format
        if is_anthropic_format(data_in):
            return messages()  # Delegate to Anthropic handler
        else:
            return chat_completions()  # Delegate to OpenAI handler

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information."""
    return jsonify({
        "name": "LLM Router",
        "description": "Flask-based router with OpenAI and Anthropic protocol support",
        "endpoints": {
            "unified": "/v1/chat",
            "openai": "/v1/chat/completions",
            "anthropic": "/v1/messages",
            "models": "/v1/models",
            "health": "/health"
        },
        "documentation": "See README.md for details"
    })




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
