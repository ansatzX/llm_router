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
MAX_ROLLBACK_RETRIES = int(os.environ.get("MAX_ROLLBACK_RETRIES", "3"))


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
    """Inject a system prompt at the beginning of messages.

    Note: This function modifies the input list in place and returns it.
    This is intentional for efficiency and documented for clarity.
    """
    if not prompt:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = prompt + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": prompt})
    return messages


def has_incomplete_mcp_tags(text: str) -> bool:
    """Check if text contains incomplete MCP XML tags.

    This indicates the model tried to call a tool but the format was incorrect.

    Args:
        text: Response text to check

    Returns:
        True if incomplete MCP tags are found
    """
    if not text:
        return False

    # MCP tag patterns (order matters - check most specific first)
    mcp_patterns = [
        "<use_mcp_tool>",
        "</use_mcp_tool>",
        "<server_name>",
        "</server_name>",
        "<tool_name>",
        "</tool_name>",
        "<arguments>",
        "</arguments>",
    ]

    return any(pattern in text for pattern in mcp_patterns)


def format_parse_errors(errors: list[str]) -> str:
    """Format parse errors into a user-friendly message for the model.

    Args:
        errors: List of error messages

    Returns:
        Formatted error message
    """
    if not errors:
        return "Unknown parsing error"

    formatted = "\n".join(f"  • {error}" for error in errors)
    return f"Tool call parsing failed with the following errors:\n{formatted}\n\nPlease fix the format and try again."


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI Chat Completions API endpoint with Rollback support."""
    try:
        data = request.get_json()

        # Log client request
        log_debug("CLIENT_REQUEST /v1/chat/completions", {
            "endpoint": "/v1/chat/completions",
            "method": "POST",
            "data": data
        })

        tools = data.get("tools", [])
        model = data.get("model", "default")
        original_messages = data.get("messages", [])

        # Create working copy of messages (for rollback modifications)
        messages = [msg.copy() for msg in original_messages]

        payload = {k: v for k, v in data.items() if k not in ("messages", "tools")}
        payload["model"] = model
        payload["messages"] = messages

        # Force stream=False when tools are provided (need full response for MCP parsing)
        if tools:
            payload["stream"] = False

        if tools:
            mcp_prompt = generate_mcp_system_prompt(tools, MCP_SERVER_NAME)
            payload["messages"] = inject_system_prompt(messages, mcp_prompt)

        # Rollback retry loop
        retry_count = 0
        llm_response = None
        parse_result = None

        while retry_count < MAX_ROLLBACK_RETRIES:
            # Make LLM request
            llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)

            choice = llm_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            response_text = message.get("content", "")

            # Also check reasoning_content for tool calls (MiroThinker puts them there)
            reasoning_text = message.get("reasoning_content", "")

            # Parse tool calls using parser module
            parse_result = parse_tool_calls(
                response_text,
                reasoning_text,
                available_tools=tools
            )

            # Log warnings
            for warning in parse_result.warnings:
                logger.warning(f"Parse warning: {warning}")

            # Log errors (don't interrupt flow)
            for error in parse_result.errors:
                logger.error(f"Parse error: {error}")

            # Check if rollback is needed
            should_retry = (
                not parse_result.success and
                parse_result.errors and
                has_incomplete_mcp_tags(response_text) and
                retry_count < MAX_ROLLBACK_RETRIES - 1
            )

            if should_retry:
                retry_count += 1

                # Log rollback
                log_debug("ROLLBACK_RETRY", {
                    "retry_count": retry_count,
                    "max_retries": MAX_ROLLBACK_RETRIES,
                    "parse_errors": parse_result.errors,
                    "response_preview": response_text[:200] if response_text else None
                })

                logger.info(f"Rollback attempt {retry_count}/{MAX_ROLLBACK_RETRIES} - parse errors detected")

                # Add error feedback to messages
                error_message = format_parse_errors(parse_result.errors)
                payload["messages"].append({
                    "role": "user",
                    "content": f"⚠️ {error_message}"
                })

                # Continue to next iteration (retry)
                continue

            # Success or non-retryable error - break loop
            break

        # Build response
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

        # Add rollback metadata if retries occurred
        if retry_count > 0:
            response["_metadata"] = {
                "rollback_attempts": retry_count,
                "rollback_success": parse_result.success
            }

        # Log summary
        usage = response.get("usage", {})
        log_debug("CLIENT_RESPONSE /v1/chat/completions", {
            "status": "success",
            "model": model,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "has_tool_calls": parse_result.success,
            "tool_calls_count": len(parse_result.tool_calls) if parse_result.success else 0,
            "rollback_attempts": retry_count if retry_count > 0 else None,
            "usage": usage,
            "response": response
        })

        return jsonify(response)

    except Exception as e:
        logger.exception("Error in chat_completions")

        # Log error
        log_debug("CLIENT_RESPONSE /v1/chat/completions ERROR", {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })

        return jsonify({
            "error": {
                "type": "server_error",
                "message": "An internal error occurred. Please try again later."
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def handle_list_models():
    """List available models from backend."""
    log_debug("CLIENT_REQUEST /v1/models", {
        "endpoint": "/v1/models",
        "method": "GET"
    })

    models = list_models(LLM_BASE_URL, LLM_API_KEY)

    log_debug("CLIENT_RESPONSE /v1/models", {
        "status": "success",
        "models_count": len(models.get("data", [])),
        "models": models
    })

    return jsonify(models)


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
