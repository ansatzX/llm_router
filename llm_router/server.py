"""
Flask server for the LLM Router.

Simplified version: OpenAI format only, MCP XML parsing for tool calls.
"""

import json
import os
import uuid
import logging
from flask import Flask, request, jsonify

from .mcp_converter import generate_mcp_system_prompt
from .llm_client import make_llm_request, list_models
from .debug_log import log_debug, set_debug_mode
from .parser import parse_tool_calls

logger = logging.getLogger(__name__)

# Configuration from environment
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
LLM_API_KEY = os.environ.get("LLM_API_KEY", None)
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5001"))
MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "tools")


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


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI Chat Completions API endpoint."""
    try:
        data = request.get_json()
        tools = data.get("tools", [])
        model = data.get("model", "default")
        messages = data.get("messages", [])

        payload = {k: v for k, v in data.items() if k not in ("messages", "tools")}
        payload["model"] = model
        payload["messages"] = messages

        # Force stream=False when tools are provided (need full response for MCP parsing)
        if tools:
            payload["stream"] = False

        if tools:
            mcp_prompt = generate_mcp_system_prompt(tools, MCP_SERVER_NAME)
            payload["messages"] = inject_system_prompt(messages, mcp_prompt)

        llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)

        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")

        # Also check reasoning_content for tool calls (MiroThinker puts them there)
        reasoning_text = message.get("reasoning_content", "")

        # Parse tool calls using new parser module
        parse_result = parse_tool_calls(response_text, reasoning_text)

        # Log warnings
        for warning in parse_result.warnings:
            logger.warning(f"Parse warning: {warning}")

        # Log errors (don't interrupt flow)
        for error in parse_result.errors:
            logger.error(f"Parse error: {error}")

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
                "finish_reason": "tool_calls" if parse_result.success else choice.get("finish_reason", "stop")
            }],
            "usage": llm_response.get("usage", {})
        }

        if parse_result.success:
            tool_calls_openai = [tc.to_openai_format() for tc in parse_result.tool_calls]
            response["choices"][0]["message"]["tool_calls"] = tool_calls_openai

        log_debug("CLIENT_RESPONSE /v1/chat/completions", response)

        return jsonify(response)

    except Exception as e:
        logger.exception("Error in chat_completions")
        return jsonify({
            "error": {
                "type": "server_error",
                "message": "An internal error occurred. Please try again later."
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def handle_list_models():
    """List available models from backend."""
    return jsonify(list_models(LLM_BASE_URL, LLM_API_KEY))


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
        "description": "API router with OpenAI format and MCP tool parsing",
        "endpoints": {
            "openai": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
