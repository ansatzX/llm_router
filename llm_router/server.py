"""Flask server for the LLM Router.

This module provides Flask endpoints for the LLM Router API, including OpenAI-compatible
chat completions with MCP XML tool parsing and rollback retry logic.
"""

import json
import logging
import os
import uuid
from typing import Any

import httpx
from flask import Flask, jsonify, request, Response, stream_with_context

from llm_router.debug_log import log_debug
from llm_router.llm_client import _get_env_float, _get_env_int, check_backend_health, list_models, make_llm_request
from llm_router.mcp_converter import generate_mcp_system_prompt
from llm_router.parser import parse_tool_calls

logger = logging.getLogger(__name__)

# Configuration from environment
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "http://localhost:8000")
LLM_API_KEY: str | None = os.environ.get("LLM_API_KEY", None)
FLASK_PORT: int = _get_env_int("FLASK_PORT", 5001)
MCP_SERVER_NAME: str = os.environ.get("MCP_SERVER_NAME", "tools")
MAX_ROLLBACK_RETRIES: int = _get_env_int("MAX_ROLLBACK_RETRIES", 3)

app = Flask(__name__)


def create_app(llm_base_url: str | None = None, llm_api_key: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        llm_base_url: Optional base URL for the LLM backend.
        llm_api_key: Optional API key for LLM backend authentication.

    Returns:
        Configured Flask application instance.
    """
    global LLM_BASE_URL, LLM_API_KEY
    if llm_base_url is not None:
        LLM_BASE_URL = llm_base_url
    if llm_api_key is not None:
        LLM_API_KEY = llm_api_key
    return app


def inject_system_prompt(messages: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """Inject a system prompt at the beginning of messages.

    Note:
        This function modifies the input list in place and returns it.
        This is intentional for efficiency and documented for clarity.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys.
        prompt: System prompt text to inject.

    Returns:
        Modified messages list with system prompt injected.
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
        text: Response text to check.

    Returns:
        True if incomplete MCP tags are found, False otherwise.
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
        errors: List of error messages.

    Returns:
        Formatted error message suitable for model retry.
    """
    if not errors:
        return "Unknown parsing error"

    formatted = "\n".join(f"  • {error}" for error in errors)
    return (
        f"[RETRY INSTRUCTION]\n"
        f"The previous tool call had parsing errors:\n{formatted}\n\n"
        f"Please regenerate the tool call with the correct format. "
        f"Do not apologize or explain - just output the corrected tool call."
    )


def convert_responses_content_item(item: dict[str, Any]) -> str:
    """Extract text from a Responses API content item.

    Args:
        item: Content item with type and text fields.

    Returns:
        Extracted text string.
    """
    if not isinstance(item, dict):
        return ""

    item_type = item.get("type")

    if item_type in ("input_text", "text"):
        return item.get("text", "")
    elif item_type == "image_url":
        # Skip images for now - would need multimodal handling
        return ""

    return ""


def convert_responses_message_to_chat(msg: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert Responses API message format to Chat Completions format.

    Responses API: {"content": [{"type": "input_text", "text": "..."}], "role": "user"}
    Or nested: {"content": [{"type": "message", "content": [...], "role": "..."}]}

    Chat Completions: {"content": "...", "role": "user"}

    Args:
        msg: Message in Responses API format.

    Returns:
        Message in Chat Completions format, or list of messages if nested.
    """
    if not isinstance(msg, dict):
        return {"role": "user", "content": ""}

    role = msg.get("role", "user")
    content = msg.get("content")
    msg_type = msg.get("type")

    # Handle string content directly
    if isinstance(content, str):
        return {"role": role, "content": content}

    # Handle content array
    if isinstance(content, list):
        # Check if this contains nested messages (type: "message")
        nested_messages = [item for item in content if isinstance(item, dict) and item.get("type") == "message"]

        if nested_messages:
            # Recursively convert nested messages
            result = []
            for nested in nested_messages:
                converted = convert_responses_message_to_chat(nested)
                if isinstance(converted, list):
                    result.extend(converted)
                else:
                    result.append(converted)
            return result

        # Regular content array - extract text
        text_parts = []
        for item in content:
            text = convert_responses_content_item(item)
            if text:
                text_parts.append(text)

        return {"role": role, "content": "\n".join(text_parts)}

    # Fallback
    return {"role": role, "content": str(content) if content else ""}


def flatten_responses_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Flatten nested Responses API messages to Chat Completions format.

    Args:
        messages: List of messages in Responses API format (potentially nested).

    Returns:
        Flat list of messages in Chat Completions format.
    """
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        converted = convert_responses_message_to_chat(msg)
        if isinstance(converted, list):
            result.extend(converted)
        else:
            result.append(converted)
    return result


@app.route('/v1/responses', methods=['POST'])
def responses_api() -> tuple[Any, int]:
    """Handle OpenAI Responses API endpoint.

    The Responses API is a simpler API that uses `input` and `instructions`
    instead of the `messages` array. This endpoint converts to Chat Completions
    format and returns in Responses API format.

    Returns:
        Tuple of (response JSON, HTTP status code).
    """
    try:
        data = request.get_json()

        # Log client request
        log_debug("CLIENT_REQUEST /v1/responses", {
            "endpoint": "/v1/responses",
            "method": "POST",
            "data": data
        })

        model = data.get("model", "default")
        instructions = data.get("instructions", "")
        input_data = data.get("input", "")
        tools = data.get("tools", [])

        # Convert Responses API format to Chat Completions format
        messages = []

        # Add system instructions if provided
        if instructions:
            messages.append({"role": "system", "content": instructions})

        # Handle input field - can be string or array of messages
        if isinstance(input_data, str) and input_data:
            messages.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            # input is an array of messages - convert and flatten
            messages = flatten_responses_messages(input_data)
            log_debug("CONVERTED_MESSAGES", {
                "original_count": len(input_data),
                "converted_count": len(messages),
                "converted_messages": messages
            })

        # Handle messages array if provided (alternative format)
        if "messages" in data:
            # Convert and flatten Responses API message format to Chat Completions format
            messages = flatten_responses_messages(data["messages"])
            log_debug("CONVERTED_MESSAGES from messages field", {
                "original_count": len(data["messages"]),
                "converted_count": len(messages),
                "converted_messages": messages
            })

        # Create working copy of messages
        messages = [msg.copy() if isinstance(msg, dict) else msg for msg in messages]

        payload = {k: v for k, v in data.items() if k not in ("messages", "tools", "input", "instructions")}
        payload["model"] = model
        payload["messages"] = messages

        # Always force stream=False for internal processing
        # (we need full response for tool parsing, then return as SSE if client requested streaming)
        client_requested_stream = data.get("stream", False)
        payload["stream"] = False

        # Inject MCP system prompt if tools provided
        if tools:
            mcp_prompt = generate_mcp_system_prompt(tools, MCP_SERVER_NAME)
            payload["messages"] = inject_system_prompt(messages, mcp_prompt)

        # Rollback retry loop
        retry_count = 0
        llm_response = None
        parse_result = None

        while retry_count < MAX_ROLLBACK_RETRIES:
            llm_response = make_llm_request(payload, LLM_BASE_URL, LLM_API_KEY)

            choice = llm_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            response_text = message.get("content", "")
            reasoning_text = message.get("reasoning_content", "")

            parse_result = parse_tool_calls(
                response_text,
                reasoning_text,
                available_tools=tools
            )

            for warning in parse_result.warnings:
                logger.warning(f"Tool call parsing: {warning}")

            if not parse_result.success:
                import re
                tool_name = None
                tool_match = re.search(r'<tool_name>([^<]+)</tool_name>', response_text)
                if tool_match:
                    tool_name = tool_match.group(1).strip()
                else:
                    json_match = re.search(r'"(?:tool_)?name"\s*:\s*"([^"]+)"', response_text)
                    if json_match:
                        tool_name = json_match.group(1).strip()

                preview_text = response_text[:200] if response_text else ""
                preview_clean = re.sub(r'<[^>]+>', '', preview_text).strip()
                preview = preview_clean[:100] if preview_clean else "(no text content)"

                for error in parse_result.errors:
                    if tool_name:
                        logger.warning(f"Tool call parsing: {error} | Tool: {tool_name} | Content: {preview}")
                    else:
                        logger.warning(f"Tool call parsing: {error} | Content: {preview}")

            should_retry = (
                not parse_result.success and
                parse_result.errors and
                has_incomplete_mcp_tags(response_text) and
                retry_count < MAX_ROLLBACK_RETRIES - 1
            )

            if should_retry:
                retry_count += 1
                log_debug("ROLLBACK_RETRY", {
                    "retry_count": retry_count,
                    "max_retries": MAX_ROLLBACK_RETRIES,
                    "parse_errors": parse_result.errors,
                    "response_preview": response_text[:200] if response_text else None
                })

                logger.info(f"Rollback attempt {retry_count}/{MAX_ROLLBACK_RETRIES} - parse errors detected")

                assistant_message = {"role": "assistant"}
                has_tool_call_in_reasoning = reasoning_text and any([
                    "<use_mcp_tool>" in reasoning_text,
                    "[TOOL_CALL]" in reasoning_text,
                    '"tool_name"' in reasoning_text,
                    '"name"' in reasoning_text and '"arguments"' in reasoning_text
                ])

                if has_tool_call_in_reasoning:
                    merged_content = f"{response_text}\n{reasoning_text}" if response_text else reasoning_text
                    assistant_message["content"] = merged_content
                else:
                    assistant_message["content"] = response_text if response_text else ""

                if assistant_message["content"]:
                    payload["messages"].append(assistant_message)

                error_message = format_parse_errors(parse_result.errors)
                payload["messages"].append({
                    "role": "user",
                    "content": error_message
                })

                continue

            break

        # Build Responses API format response
        output_text = response_text if not parse_result.success else None

        # Convert usage format: prompt_tokens -> input_tokens, completion_tokens -> output_tokens
        raw_usage = llm_response.get("usage", {})
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0)
        }

        response = {
            "id": llm_response.get("id", f"resp-{uuid.uuid4().hex[:8]}"),
            "object": "response",
            "created": llm_response.get("created", 0),
            "model": model,
            "output_text": output_text,
            "usage": usage,
            "status": "completed"
        }

        # Add tool_calls if parsing succeeded
        if parse_result.success:
            response["tool_calls"] = [tc.to_openai_format() for tc in parse_result.tool_calls]

        # Add rollback metadata if retries occurred
        if retry_count > 0:
            response["_metadata"] = {
                "rollback_attempts": retry_count,
                "rollback_success": parse_result.success
            }

        # Log summary
        usage = response.get("usage", {})
        log_debug("CLIENT_RESPONSE /v1/responses", {
            "status": "success",
            "model": model,
            "has_tool_calls": parse_result.success,
            "tool_calls_count": len(parse_result.tool_calls) if parse_result.success else 0,
            "rollback_attempts": retry_count if retry_count > 0 else None,
            "usage": usage,
            "response": response
        })

        # Check if client requested streaming - return as SSE
        if client_requested_stream:
            # Return SSE streaming response in Codex/OpenAI format
            # Format based on codex fixture: event: <type>\ndata: <json>\n\n
            def generate_sse():
                # 1. Send response.created event (minimal format)
                created_event = {
                    "type": "response.created",
                    "response": {"id": response["id"]}
                }
                yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"

                # 2. Send response.output_item.done with the message content
                if output_text:
                    output_item_event = {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": output_text}]
                        }
                    }
                    yield f"event: response.output_item.done\ndata: {json.dumps(output_item_event)}\n\n"

                # 3. Send tool calls if present (as function_call items)
                if parse_result.success and parse_result.tool_calls:
                    for tc in parse_result.tool_calls:
                        tc_data = tc.to_openai_format()
                        tool_call_event = {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "function_call",
                                "id": tc_data.get("id", ""),
                                "call_id": tc_data.get("id", ""),
                                "name": tc_data["function"]["name"],
                                "arguments": tc_data["function"]["arguments"]
                            }
                        }
                        yield f"event: response.output_item.done\ndata: {json.dumps(tool_call_event)}\n\n"

                # 4. Send response.completed event (minimal format)
                completed_event = {
                    "type": "response.completed",
                    "response": {
                        "id": response["id"],
                        "usage": {
                            "input_tokens": usage["input_tokens"],
                            "output_tokens": usage["output_tokens"],
                            "total_tokens": usage["total_tokens"]
                        }
                    }
                }
                yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"

            return Response(
                stream_with_context(generate_sse()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

        return jsonify(response)

        return jsonify(response)

    except Exception as e:
        logger.exception("Error in responses_api")

        log_debug("CLIENT_RESPONSE /v1/responses ERROR", {
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


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions() -> tuple[Any, int]:
    """Handle OpenAI Chat Completions API endpoint with rollback support.

    This endpoint processes chat completion requests, injects MCP system prompts
    when tools are provided, and implements rollback retry logic for failed
    tool call parsing.

    Returns:
        Tuple of (response JSON, HTTP status code).
    """
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
                logger.warning(f"Tool call parsing: {warning}")

            # Log parse results (informational, not errors)
            if not parse_result.success:
                # Extract tool_name if present (for better debugging)
                import re
                tool_name = None

                # Try to extract from XML format
                tool_match = re.search(r'<tool_name>([^<]+)</tool_name>', response_text)
                if tool_match:
                    tool_name = tool_match.group(1).strip()
                else:
                    # Try JSON format
                    json_match = re.search(r'"(?:tool_)?name"\s*:\s*"([^"]+)"', response_text)
                    if json_match:
                        tool_name = json_match.group(1).strip()

                # Show meaningful text content, not XML tags
                preview_text = response_text[:200] if response_text else ""
                preview_clean = re.sub(r'<[^>]+>', '', preview_text).strip()
                preview = preview_clean[:100] if preview_clean else "(no text content)"

                # Format log message with tool_name if found
                for error in parse_result.errors:
                    if tool_name:
                        logger.warning(
                            f"Tool call parsing: {error} | Tool: {tool_name} | Content: {preview}"
                        )
                    else:
                        logger.warning(
                            f"Tool call parsing: {error} | Content: {preview}"
                        )

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

                # Add assistant's failed response to conversation history
                # Strategy: Include content, and reasoning_content only if it contains tool calls
                # This avoids confusing the model with its own reasoning, while ensuring
                # it can see tool calls if they were placed in reasoning_content
                assistant_message = {"role": "assistant"}

                # Check if reasoning_content contains potential tool call markers
                has_tool_call_in_reasoning = reasoning_text and any([
                    "<use_mcp_tool>" in reasoning_text,
                    "[TOOL_CALL]" in reasoning_text,
                    '"tool_name"' in reasoning_text,
                    '"name"' in reasoning_text and '"arguments"' in reasoning_text
                ])

                if has_tool_call_in_reasoning:
                    # Merge both fields so model can see the complete tool call
                    merged_content = f"{response_text}\n{reasoning_text}" if response_text else reasoning_text
                    assistant_message["content"] = merged_content
                else:
                    # Only include content
                    assistant_message["content"] = response_text if response_text else ""

                # Only add if there's actual content
                if assistant_message["content"]:
                    payload["messages"].append(assistant_message)

                # Add error feedback as user message
                error_message = format_parse_errors(parse_result.errors)
                payload["messages"].append({
                    "role": "user",
                    "content": error_message
                })

                # Continue to next iteration (retry)
                continue

            # Success or non-retryable error - break loop
            break

        # Build response
        # According to OpenAI API spec, content should be null when tool_calls exist
        response = {
            "id": llm_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            "object": "chat.completion",
            "created": llm_response.get("created", 0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if parse_result.success else response_text,
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
def handle_list_models() -> Any:
    """List available models from backend.

    Returns:
        JSON response with list of available models.
    """
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
def health_check() -> tuple[Any, int]:
    """Health check endpoint with backend verification.

    Checks if backend LLM server is responding and returns appropriate HTTP status code.
    Returns 200 if backend is healthy, 503 if unhealthy.

    Returns:
        Tuple of (JSON response, HTTP status code).
    """
    # Get health check timeout from environment
    health_timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)

    # Check backend health
    backend_result = check_backend_health(
        LLM_BASE_URL,
        timeout=health_timeout,
        api_key=LLM_API_KEY
    )

    # Build response
    is_healthy = backend_result["healthy"]

    response = {
        "status": "healthy" if is_healthy else "unhealthy",
        "router": {
            "status": "ok"
        },
        "backend": {
            "status": "healthy" if is_healthy else "unhealthy",
            "url": LLM_BASE_URL,
            "latency_ms": backend_result.get("latency_ms"),
            "error": backend_result.get("error")
        }
    }

    # Log health check result
    log_debug("HEALTH_CHECK", {
        "healthy": is_healthy,
        "backend_url": LLM_BASE_URL,
        "latency_ms": backend_result.get("latency_ms"),
        "error": backend_result.get("error")
    })

    status_code = 200 if is_healthy else 503
    return jsonify(response), status_code


@app.route('/liveness', methods=['GET'])
def liveness() -> tuple[Any, int]:
    """Kubernetes liveness probe endpoint.

    Checks if the router process is alive (does NOT check backend).
    Kubernetes uses this to restart the container if it fails.

    Returns:
        Tuple of (JSON response, HTTP status code) - always 200 if router is running.
    """
    return jsonify({"status": "alive"}), 200


@app.route('/readiness', methods=['GET'])
def readiness() -> tuple[Any, int]:
    """Kubernetes readiness probe endpoint with backend verification.

    Checks if backend is ready to serve requests.
    Kubernetes uses this to determine if pod should receive traffic.

    Returns:
        Tuple of (JSON response, HTTP status code).
        200 if backend is ready, 503 if not ready.
    """
    # Get health check timeout from environment
    health_timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)

    # Check backend health
    backend_result = check_backend_health(
        LLM_BASE_URL,
        timeout=health_timeout,
        api_key=LLM_API_KEY
    )

    is_ready = backend_result["healthy"]

    response = {
        "status": "ready" if is_ready else "not_ready",
        "backend": {
            "url": LLM_BASE_URL,
            "latency_ms": backend_result.get("latency_ms"),
            "error": backend_result.get("error")
        }
    }

    # Log readiness check
    log_debug("READINESS_CHECK", {
        "ready": is_ready,
        "backend_url": LLM_BASE_URL,
        "latency_ms": backend_result.get("latency_ms"),
        "error": backend_result.get("error")
    })

    status_code = 200 if is_ready else 503
    return jsonify(response), status_code


@app.route('/metrics', methods=['GET'])
def metrics() -> tuple[Any, int]:
    """Proxy backend metrics endpoint for deployment platform health checks.

    Deployment platforms poll /metrics to verify model readiness.
    This endpoint forwards the request to SGLang backend and returns
    the metrics with appropriate HTTP status code.

    Returns:
        Tuple of (metrics response, HTTP status code).
        200 with backend metrics if healthy, 503 if backend unreachable.
    """
    # Get health check timeout from environment
    health_timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)

    # Build backend metrics URL
    base_url = LLM_BASE_URL.rstrip("/")
    metrics_url = f"{base_url}/metrics"

    try:
        with httpx.Client(timeout=health_timeout) as client:
            response = client.get(metrics_url)

        if response.status_code == 200:
            # Log successful metrics request
            log_debug("METRICS_CHECK", {
                "healthy": True,
                "backend_url": LLM_BASE_URL,
                "status_code": 200
            })

            # Forward backend response
            return Response(
                response.content,
                status=200,
                mimetype='text/plain'  # Prometheus metrics are text/plain
            )
        else:
            # Backend returned error
            log_debug("METRICS_CHECK", {
                "healthy": False,
                "backend_url": LLM_BASE_URL,
                "status_code": response.status_code
            })

            return jsonify({
                "error": f"Backend returned HTTP {response.status_code}",
                "backend_url": LLM_BASE_URL
            }), 503

    except httpx.TimeoutException:
        log_debug("METRICS_CHECK", {
            "healthy": False,
            "backend_url": LLM_BASE_URL,
            "error": f"Timeout after {health_timeout}s"
        })

        return jsonify({
            "error": f"Backend timeout after {health_timeout}s",
            "backend_url": LLM_BASE_URL
        }), 503

    except httpx.ConnectError:
        log_debug("METRICS_CHECK", {
            "healthy": False,
            "backend_url": LLM_BASE_URL,
            "error": "Connection refused"
        })

        return jsonify({
            "error": "Backend connection refused",
            "backend_url": LLM_BASE_URL
        }), 503

    except Exception as e:
        logger.exception("Error in metrics endpoint")

        log_debug("METRICS_CHECK", {
            "healthy": False,
            "backend_url": LLM_BASE_URL,
            "error": str(e)
        })

        return jsonify({
            "error": f"Backend unreachable: {str(e)}",
            "backend_url": LLM_BASE_URL
        }), 503


@app.route('/', methods=['GET'])
def index() -> Any:
    """Root endpoint with API information.

    Returns:
        JSON response with API name, description, and available endpoints.
    """
    return jsonify({
        "name": "LLM Router",
        "description": "API router with OpenAI format and MCP tool parsing",
        "endpoints": {
            "openai": "/v1/chat/completions",
            "responses": "/v1/responses",
            "models": "/v1/models",
            "health": "/health",
            "liveness": "/liveness",
            "readiness": "/readiness",
            "metrics": "/metrics"
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
