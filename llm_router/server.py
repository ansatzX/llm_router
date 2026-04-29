"""LLM Router — TOML-configured model router with Responses session management.

Routes:
  /v1/chat/completions — OpenAI Chat Completions API
  /v1/responses        — OpenAI Responses API (converts to Chat internally)
  /v1/models           — Model list proxy
  /health, /liveness, /readiness, /metrics — Health probes
"""

import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from flask import Flask, Response, jsonify, request, stream_with_context

from llm_router.config import RouterConfig
from llm_router.debug_log import log_debug
from llm_router.deepseek import DeepSeekChatAdapter
from llm_router.llm_client import (
    _get_env_float,
    check_backend_health,
    list_models,
    make_llm_request,
)
from llm_router.mirothinker import MiroThinkerMCPAdapter
from llm_router.session_store import SessionStore

logger = logging.getLogger(__name__)

# ── Globals (set on app creation) ───────────────────────────────────────────

_config: RouterConfig | None = None
_sessions: SessionStore | None = None
_deepseek_adapter = DeepSeekChatAdapter()
_mirothinker_adapter = MiroThinkerMCPAdapter()

app = Flask(__name__)


def create_app(config_path: str | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_path: Path to router.toml. Searches default locations if None.
    """
    global _config, _sessions

    _config = RouterConfig.from_toml(config_path) if config_path else RouterConfig.find_and_load()

    _sessions = SessionStore(ttl_seconds=_config.session_ttl_seconds)
    _deepseek_adapter.reset()

    logger.info("LLM Router loaded: %d upstreams, %d routes",
                len(_config.upstreams), len(_config.routes))
    return app


def _resolve(model: str) -> tuple[str, str, str, str]:
    """Resolve model name → (model_type, upstream_name, upstream_base_url).

    model_type: "mcp_first" | "responses" | "chat"
    """
    model_type, upstream = _config.resolve(model)
    upstream_name = next(
        name for name, cfg in _config.upstreams.items() if cfg is upstream
    )
    return model_type, upstream_name, upstream.base_url, upstream.resolve_api_key()


def _is_deepseek_upstream(upstream_name: str, base_url: str) -> bool:
    return upstream_name == "deepseek" or "api.deepseek.com" in base_url


def convert_chat_tool_to_responses(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Chat Completions tool format → Responses API format.

    Chat: {"type": "function", "function": {"name": ..., "parameters": ...}}
    Responses: {"type": "function", "name": ..., "parameters": ...}
    """
    if tool.get("type") == "function":
        if "function" in tool:
            func = tool["function"]
            return {
                "type": "function",
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
            }
        elif "name" in tool:
            return tool
    return tool


def convert_responses_tool_to_chat(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert Responses API function tool format to Chat Completions format."""
    return _deepseek_adapter.response_tool_to_chat(tool)


# ── Tool / Rollback Helpers ────────────────────────────────────────────────


# ── SSE Helpers ─────────────────────────────────────────────────────────────


def _build_sse_response(
    response_id: str,
    output_items: list[dict[str, Any]],
    usage: dict[str, int],
) -> Response:
    """Build SSE streaming response in Responses API format.

    OpenAI Responses SSE format:
      event: <type>
      data: <json>
      <blank line>
    """
    def generate_sse():
        # 1. response.created
        created = {
            "type": "response.created",
            "response": {"id": response_id},
        }
        yield f"event: response.created\ndata: {json.dumps(created)}\n\n"

        # 2. Output items
        for item in output_items:
            item_event = {
                "type": "response.output_item.done",
                "item": item,
            }
            yield f"event: response.output_item.done\ndata: {json.dumps(item_event)}\n\n"

        # 3. response.completed
        completed = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "usage": {
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                },
            },
        }
        yield f"event: response.completed\ndata: {json.dumps(completed)}\n\n"

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Core LLM Logic (shared by both endpoints) ─────────────────────────────


def _run_llm_with_rollback(
    payload: dict[str, Any],
    base_url: str,
    api_key: str,
    tools: list[dict[str, Any]],
    inject_mcp: bool,
) -> tuple[dict[str, Any], Any, int, str | None]:
    """Make LLM request with optional MCP tool parsing + rollback.

    Returns:
        (llm_response, parse_result, retry_count, response_text)
    """
    max_retries = _config.max_rollback_retries

    # Inject MCP prompt for mcp_first models
    if inject_mcp and tools:
        _mirothinker_adapter.prepare_payload(
            payload,
            tools,
            _config.mcp_server_name,
        )

    retry_count = 0
    llm_response = None
    parse_result = None
    response_text = ""

    while retry_count < max_retries:
        llm_response = make_llm_request(payload, base_url, api_key)

        choice = llm_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        response_text = message.get("content", "")
        reasoning_text = message.get("reasoning_content", "")

        if inject_mcp and tools:
            parse_result = _mirothinker_adapter.parse_message(
                response_text,
                reasoning_text,
                tools,
            )

            for warning in parse_result.warnings:
                logger.warning(f"Tool parsing: {warning}")

            if not parse_result.success:
                for error in parse_result.errors:
                    logger.warning(f"Tool parsing error: {error}")

            should_retry = _mirothinker_adapter.should_retry(
                parse_result,
                response_text,
                retry_count,
                max_retries,
            )

            if should_retry:
                retry_count += 1
                logger.info(
                    "Rollback %d/%d: parse errors detected",
                    retry_count, max_retries,
                )

                _mirothinker_adapter.append_retry_feedback(
                    payload,
                    response_text,
                    parse_result.errors,
                )
                continue

        break

    return llm_response, parse_result, retry_count, response_text


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.route("/v1/responses", methods=["POST"])
def responses_api() -> tuple[Any, int]:
    """OpenAI Responses API → Chat Completions translation with session state.

    For "responses" route type: accumulates items via session store,
    converts to Chat format, forwards to upstream, and returns SSE.

    For "mcp_first" route type: same, but injects MCP XML tool prompt.
    """
    try:
        data = request.get_json()
        log_debug("CLIENT_REQUEST /v1/responses", {
            "endpoint": "/v1/responses",
            "method": "POST", "data": data,
        })

        model = data.get("model", "default")
        model_type, upstream_name, base_url, api_key = _resolve(model)
        is_deepseek = _is_deepseek_upstream(upstream_name, base_url)

        instructions = data.get("instructions", "")
        input_data = data.get("input", [])
        tools_raw = data.get("tools", [])
        previous_response_id = data.get("previous_response_id")
        client_requested_stream = data.get("stream", False)
        tool_type_map = _deepseek_adapter.tool_type_map(tools_raw)

        # Convert tools to Chat format for internal use
        tools = [convert_chat_tool_to_responses(t) for t in tools_raw]

        # ── Session state management ──
        inject_mcp = (model_type == "mcp_first")

        if model_type == "responses":
            # Stateful: get or create session
            session = _sessions.get_or_create(previous_response_id, model)

            # Add new input items to accumulated history
            if isinstance(input_data, list) and input_data:
                _sessions.add_items(session, input_data)

            # Convert full history to chat messages
            messages = session.to_chat_messages(
                _deepseek_adapter.flatten_response_items,
            )
        else:
            # Stateless (chat / mcp_first): convert input directly
            messages = []
            if isinstance(input_data, str) and input_data:
                messages.append({"role": "user", "content": input_data})
            elif isinstance(input_data, list):
                messages = _deepseek_adapter.flatten_response_items(input_data)

        # Prepend instructions as system message
        if instructions:
            messages.insert(0, {"role": "system", "content": instructions})

        # Build payload
        payload = {
            k: v for k, v in data.items()
            if k not in ("messages", "tools", "input", "instructions",
                         "previous_response_id", "stream", "store", "include")
        }
        payload["model"] = model
        payload["messages"] = messages
        payload["stream"] = False  # Always non-streaming internally
        if tools_raw and not inject_mcp:
            chat_tools = (
                _deepseek_adapter.responses_tools_to_chat(tools_raw)
                if is_deepseek
                else [convert_responses_tool_to_chat(t) for t in tools_raw]
            )
            if chat_tools:
                payload["tools"] = chat_tools

        # ── Run LLM ──
        llm_response, parse_result, retry_count, response_text = (
            _run_llm_with_rollback(payload, base_url, api_key, tools, inject_mcp)
        )

        # ── Build response ──
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        choice = llm_response.get("choices", [{}])[0]
        response_message = choice.get("message", {})
        output_items, output_text, native_tool_calls = (
            _deepseek_adapter.chat_response_to_output_items(
                response_message,
                tool_type_map,
            )
        )

        if model_type == "responses":
            for item in output_items:
                _sessions.add_output_item(session, item)
            _sessions.register_response_id(session, response_id)

        # Convert usage
        raw_usage = llm_response.get("usage", {})
        usage = {
            "input_tokens": raw_usage.get("prompt_tokens", 0),
            "output_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }

        # Extract tool calls
        tool_calls_list = []
        if inject_mcp and parse_result and parse_result.success:
            output_items, tool_calls_list = (
                _mirothinker_adapter.to_responses_tool_outputs(
                    parse_result,
                )
            )
        elif native_tool_calls:
            tool_calls_list = native_tool_calls

        has_tool_output = bool(tool_calls_list)
        output_text = None if has_tool_output else output_text

        response = {
            "id": response_id,
            "object": "response",
            "created": llm_response.get("created", int(time.time())),
            "model": model,
            "output": output_items,
            "output_text": output_text,
            "usage": usage,
            "status": "completed",
        }
        if retry_count > 0:
            response["_metadata"] = {
                "rollback_attempts": retry_count,
                "rollback_success": bool(inject_mcp and parse_result and parse_result.success),
            }
        if tool_calls_list:
            response["tool_calls"] = tool_calls_list

        log_debug("CLIENT_RESPONSE /v1/responses", {
            "status": "success", "model": model, "model_type": model_type,
            "has_tool_calls": bool(tool_calls_list),
            "tool_calls_count": len(tool_calls_list),
            "rollback_attempts": retry_count if retry_count else None,
            "session_items": len(session.items) if model_type == "responses" else None,
            "usage": usage,
        })

        if client_requested_stream:
            return _build_sse_response(response_id, output_items, usage)

        return jsonify(response), 200

    except Exception:
        logger.exception("Error in responses_api")
        return jsonify({
            "error": {"type": "server_error",
                      "message": "An internal error occurred."}
        }), 500


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> tuple[Any, int]:
    """OpenAI Chat Completions endpoint with routing and MCP support."""
    try:
        data = request.get_json()
        log_debug("CLIENT_REQUEST /v1/chat/completions", {
            "endpoint": "/v1/chat/completions",
            "method": "POST", "data": data,
        })

        model = data.get("model", "default")
        model_type, upstream_name, base_url, api_key = _resolve(model)
        tools = data.get("tools", [])
        messages = [m.copy() for m in data.get("messages", [])]

        payload = {
            k: v for k, v in data.items()
            if k not in ("messages", "tools")
        }
        payload["model"] = model
        payload["messages"] = messages
        payload["stream"] = False  # Always non-streaming internally

        inject_mcp = (model_type == "mcp_first")
        llm_response, parse_result, retry_count, response_text = (
            _run_llm_with_rollback(payload, base_url, api_key, tools, inject_mcp)
        )

        # Build response
        choice = llm_response.get("choices", [{}])[0]
        response = {
            "id": llm_response.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            "object": "chat.completion",
            "created": llm_response.get("created", 0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if (inject_mcp and parse_result and parse_result.success)
                    else response_text,
                },
                "finish_reason": (
                    "tool_calls"
                    if (inject_mcp and parse_result and parse_result.success)
                    else choice.get("finish_reason", "stop")
                ),
            }],
            "usage": llm_response.get("usage", {}),
        }

        if inject_mcp and parse_result and parse_result.success:
            tc_list = _mirothinker_adapter.to_openai_tool_calls(parse_result)
            response["choices"][0]["message"]["tool_calls"] = tc_list

        if retry_count > 0:
            response["_metadata"] = {
                "rollback_attempts": retry_count,
                "rollback_success": bool(parse_result and parse_result.success),
            }

        log_debug("CLIENT_RESPONSE /v1/chat/completions", {
            "status": "success", "model": model, "model_type": model_type,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "has_tool_calls": bool(response["choices"][0]["message"].get("tool_calls")),
            "rollback_attempts": retry_count if retry_count else None,
            "usage": llm_response.get("usage", {}),
        })

        return jsonify(response), 200

    except Exception:
        logger.exception("Error in chat_completions")
        return jsonify({
            "error": {"type": "server_error",
                      "message": "An internal error occurred."}
        }), 500


@app.route("/v1/models", methods=["GET"])
def handle_list_models() -> Any:
    """List models from the default upstream."""
    upstream = _config.upstreams[_config.default_upstream]
    models = list_models(upstream.base_url, upstream.resolve_api_key() or os.environ.get("LLM_API_KEY"))
    return jsonify(models)


@app.route("/health", methods=["GET"])
def health_check() -> tuple[Any, int]:
    upstream = _config.upstreams[_config.default_upstream]
    health_timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)
    result = check_backend_health(
        upstream.base_url, timeout=health_timeout,
        api_key=upstream.resolve_api_key(),
    )
    is_healthy = result["healthy"]
    return jsonify({
        "status": "healthy" if is_healthy else "unhealthy",
        "router": {"status": "ok"},
        "backend": {
            "status": "healthy" if is_healthy else "unhealthy",
            "url": upstream.base_url,
            "latency_ms": result.get("latency_ms"),
            "error": result.get("error"),
        },
    }), 200 if is_healthy else 503


@app.route("/liveness", methods=["GET"])
def liveness() -> tuple[Any, int]:
    return jsonify({"status": "alive"}), 200


@app.route("/readiness", methods=["GET"])
def readiness() -> tuple[Any, int]:
    upstream = _config.upstreams[_config.default_upstream]
    health_timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)
    result = check_backend_health(
        upstream.base_url, timeout=health_timeout,
        api_key=upstream.resolve_api_key(),
    )
    is_ready = result["healthy"]
    return jsonify({
        "status": "ready" if is_ready else "not_ready",
        "backend": {"url": upstream.base_url,
                     "latency_ms": result.get("latency_ms"),
                     "error": result.get("error")},
    }), 200 if is_ready else 503


@app.route("/metrics", methods=["GET"])
def metrics() -> tuple[Any, int]:
    upstream = _config.upstreams[_config.default_upstream]
    health_timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)
    base_url = upstream.base_url.rstrip("/")
    metrics_url = f"{base_url}/metrics"
    try:
        with httpx.Client(timeout=health_timeout) as client:
            resp = client.get(metrics_url)
        if resp.status_code == 200:
            return Response(resp.content, status=200, mimetype="text/plain")
        return jsonify({"error": f"Backend returned HTTP {resp.status_code}"}), 503
    except httpx.TimeoutException:
        return jsonify({"error": f"Backend timeout after {health_timeout}s"}), 503
    except httpx.ConnectError:
        return jsonify({"error": "Backend connection refused"}), 503
    except Exception as e:
        return jsonify({"error": f"Backend unreachable: {e}"}), 503


@app.route("/", methods=["GET"])
def index() -> Any:
    return jsonify({
        "name": "LLM Router",
        "description": "TOML-configured model router with Responses session management",
        "endpoints": {
            "responses": "/v1/responses",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "liveness": "/liveness",
            "readiness": "/readiness",
            "metrics": "/metrics",
        },
    })


if __name__ == "__main__":
    _config = RouterConfig.find_and_load()
    _sessions = SessionStore(ttl_seconds=_config.session_ttl_seconds)
    app.run(host=_config.server_host, port=_config.server_port, debug=False)
