"""LLM Router — TOML-configured model router with Responses session management.

Routes:
  /v1/chat/completions — OpenAI Chat Completions API
  /v1/responses        — OpenAI Responses API (converts to Chat internally)
  /v1/models           — Model list proxy
  /health, /liveness, /readiness, /metrics — Health probes
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from flask import Flask, Response, jsonify, request, stream_with_context

from llm_router.codex_recovery import (
    _append_plan_mode_proposed_plan_feedback,
    _append_plan_mode_retry_feedback,
    _extract_collaboration_mode,
    _plan_mode_mutation_violation,
    _plan_mode_should_retry_with_request_user_input,
    _responses_request_diagnostics,
    _responses_response_diagnostics,
    _validate_plan_mode_output_items,
)
from llm_router.config import RouterConfig
from llm_router.debug_log import log_debug
from llm_router.deepseek import DeepSeekChatAdapter
from llm_router.llm_client import (
    LLMRequestError,
    ResponsesPassthroughError,
    _get_env_float,
    check_backend_health,
    list_models,
    make_llm_request,
    make_responses_request,
)
from llm_router.mirothinker import MiroThinkerMCPAdapter
from llm_router.openai_chat import OpenAIChatAdapter
from llm_router.provider_errors import _llm_request_error_body
from llm_router.responses_state import (
    ResponsesStateError,
    ResponsesStateMachine,
    ResponsesTurn,
    iter_sse_events,
)
from llm_router.responses_state.tools import (
    _normalize_responses_tools,
    convert_responses_tool_to_chat,
)
from llm_router.responses_state.usage import _responses_usage_from_provider
from llm_router.session_store import SessionStore

logger = logging.getLogger(__name__)

_config: RouterConfig | None = None
_sessions: SessionStore | None = None
_deepseek_adapter = DeepSeekChatAdapter()
_mirothinker_adapter = MiroThinkerMCPAdapter()

app = Flask(__name__)

_MEMORY_MODEL_ALIASES = {
    "gpt-5.4-mini": "deepseek-v4-flash",
    "gpt-5.4": "deepseek-v4-pro",
}
_MEMORY_PROMPT_MARKERS = (
    "## Memory Writing Agent: Phase 1 (Single Rollout)",
    "Analyze this rollout and produce JSON with `raw_memory`, `rollout_summary`, and `rollout_slug`",
    "## Memory Writing Agent: Phase 2 (Consolidation)",
    "Consolidate Codex memories in:",
)


@dataclass
class _ResponsesTurnContext:
    state_machine: ResponsesStateMachine
    turn: ResponsesTurn
    model_type: str
    upstream_name: str
    base_url: str
    api_key: str
    upstream_model: str
    is_deepseek: bool
    chat_adapter: DeepSeekChatAdapter
    tool_type_map: dict[str, str]
    inject_mcp: bool
    messages: list[dict[str, Any]]
    collaboration_mode: str | None


@dataclass
class _ResponsesProviderResult:
    llm_response: dict[str, Any]
    parse_result: Any
    retry_count: int
    response_message: dict[str, Any]
    output_items: list[dict[str, Any]]
    output_text: str | None
    tool_calls_list: list[dict[str, Any]]
    usage: dict[str, Any]


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


def _request_is_memory_workload(
    model: str,
    data: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> bool:
    """Detect Codex background memories requests without relying on model slug alone."""
    if model not in _MEMORY_MODEL_ALIASES:
        return False

    memgen_header = request.headers.get("x-openai-memgen-request", "")
    if memgen_header.lower() == "true":
        return True

    subagent_header = request.headers.get("x-openai-subagent", "")
    if subagent_header == "memory_consolidation":
        return True

    haystacks: list[str] = []
    if isinstance(data, dict):
        instructions = data.get("instructions")
        if isinstance(instructions, str):
            haystacks.append(instructions)
    if messages:
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                haystacks.append(content)

    return any(marker in text for text in haystacks for marker in _MEMORY_PROMPT_MARKERS)


def _resolve(
    model: str,
    data: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[str, str, str, str, str]:
    """Resolve model name → routing info.

    model_type: "mcp_first" | "responses" | "responses_chat"
    """
    model_type, upstream, upstream_model = _config.resolve_request(model)
    upstream_name = next(
        name for name, cfg in _config.upstreams.items() if cfg is upstream
    )
    if (
        _request_is_memory_workload(model, data, messages)
        and "deepseek" in _config.upstreams
    ):
        deepseek = _config.upstreams["deepseek"]
        rewritten_model = _MEMORY_MODEL_ALIASES[model]
        log_debug("MEMORY_MODEL_OVERRIDE", {
            "requested_model": model,
            "upstream_name": "deepseek",
            "upstream_model": rewritten_model,
            "memgen_header": request.headers.get("x-openai-memgen-request"),
            "subagent_header": request.headers.get("x-openai-subagent"),
        })
        return (
            "responses_chat",
            "deepseek",
            deepseek.base_url,
            deepseek.resolve_api_key(),
            rewritten_model,
        )
    return (
        model_type,
        upstream_name,
        upstream.base_url,
        upstream.resolve_api_key(),
        upstream_model,
    )


def _is_deepseek_upstream(upstream_name: str, base_url: str) -> bool:
    return upstream_name == "deepseek" or "api.deepseek.com" in base_url


def _is_official_deepseek_base_url(base_url: str) -> bool:
    return urlparse(base_url.rstrip("/")).netloc.lower() == "api.deepseek.com"


def _chat_adapter_for(upstream_name: str, base_url: str) -> DeepSeekChatAdapter:
    if _is_deepseek_upstream(upstream_name, base_url):
        return DeepSeekChatAdapter()
    return OpenAIChatAdapter()


def _apply_provider_defaults(payload: dict[str, Any], model_type: str) -> dict[str, Any]:
    """Apply provider-specific defaults after adapter filtering."""
    if model_type == "mcp_first" and "repetition_penalty" not in payload:
        payload["repetition_penalty"] = _get_env_float(
            "DEFAULT_REPETITION_PENALTY",
            1.05,
        )
    return payload


def _tool_call_ids_from_response_items(items: list[dict[str, Any]]) -> set[str]:
    call_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"function_call", "custom_tool_call"}:
            continue
        call_id = item.get("call_id") or item.get("id")
        if isinstance(call_id, str) and call_id:
            call_ids.add(call_id)
    return call_ids


def _merge_deepseek_provider_states(
    *states: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    merged: dict[str, str] = {}
    for state in states:
        if not isinstance(state, dict):
            continue
        reasoning_by_call_id = state.get("reasoning_by_call_id", {})
        if not isinstance(reasoning_by_call_id, dict):
            continue
        for call_id, reasoning_content in reasoning_by_call_id.items():
            if call_id and reasoning_content:
                merged[str(call_id)] = str(reasoning_content)
    return {"reasoning_by_call_id": merged}


def _build_sse_response(
    response_id: str,
    output_items: list[dict[str, Any]],
    usage: dict[str, Any],
) -> Response:
    """Build SSE streaming response in Responses API format."""
    return Response(
        stream_with_context(iter_sse_events(response_id, output_items, usage)),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


def _handle_responses_passthrough(
    data: dict[str, Any],
    model: str,
    model_type: str,
    upstream_name: str,
    base_url: str,
    api_key: str,
    upstream_model: str,
    tools_raw: list[dict[str, Any]],
    normalized_tools: list[dict[str, Any]],
    client_requested_stream: bool,
) -> tuple[Any, int]:
    passthrough_payload = dict(data)
    passthrough_payload["model"] = upstream_model
    passthrough_payload["stream"] = False
    if tools_raw:
        passthrough_payload["tools"] = normalized_tools
    try:
        passthrough_response = make_responses_request(
            passthrough_payload,
            base_url,
            api_key,
        )
    except ResponsesPassthroughError as exc:
        log_debug("RESPONSES_PASSTHROUGH_ERROR", {
            "model": model,
            "upstream": upstream_name,
            "error": str(exc),
        })
        return jsonify({
            "error": {
                "type": "provider_error",
                "message": str(exc),
            },
        }), exc.status_code
    except Exception as exc:
        log_debug("RESPONSES_PASSTHROUGH_ERROR", {
            "model": model,
            "upstream": upstream_name,
            "error": str(exc),
        })
        return jsonify({
            "error": {
                "type": "provider_error",
                "message": str(exc),
            },
        }), 502

    response = dict(passthrough_response)
    response["model"] = model
    usage = _responses_usage_from_provider(response.get("usage", {}))
    response["usage"] = usage
    log_debug("CLIENT_RESPONSE /v1/responses", {
        "status": "passthrough_success",
        "model": model,
        "model_type": model_type,
        "upstream": upstream_name,
        "usage": usage,
    })
    if client_requested_stream:
        return _build_sse_response(
            response["id"],
            response.get("output", []),
            usage,
        )
    return jsonify(response), 200


def _load_deepseek_provider_state(
    turn: ResponsesTurn,
    chat_adapter: DeepSeekChatAdapter,
) -> None:
    call_ids = _tool_call_ids_from_response_items(
        [*turn.session.items, *turn.input_items],
    )
    recovered_provider_state = _sessions.provider_state_for_call_ids(
        "deepseek",
        call_ids,
    )
    session_provider_state = turn.session.provider_state.get("deepseek")
    chat_adapter.load_provider_state(
        _merge_deepseek_provider_states(
            recovered_provider_state,
            session_provider_state,
        ),
    )
    recovered_call_ids = sorted(
        set(
            recovered_provider_state.get(
                "reasoning_by_call_id",
                {},
            )
        )
    )
    if recovered_call_ids:
        log_debug("DEEPSEEK_PROVIDER_STATE_RECOVERY", {
            "recovered_call_ids": recovered_call_ids,
            "session_had_provider_state": bool(session_provider_state),
        })


def _prepare_router_owned_responses_context(
    data: dict[str, Any],
    model: str,
    instructions: str,
    tools_raw: list[dict[str, Any]],
) -> _ResponsesTurnContext:
    state_machine = ResponsesStateMachine(_sessions)
    turn = state_machine.ingest_request(data, model)
    provisional_adapter = OpenAIChatAdapter()
    messages = turn.to_chat_messages(provisional_adapter.flatten_response_items)
    model_type, upstream_name, base_url, api_key, upstream_model = _resolve(
        model,
        data,
        messages,
    )
    is_deepseek = _is_deepseek_upstream(upstream_name, base_url)
    chat_adapter = _chat_adapter_for(upstream_name, base_url)
    tool_type_map = chat_adapter.tool_type_map(tools_raw)
    inject_mcp = (model_type == "mcp_first")
    if is_deepseek:
        _load_deepseek_provider_state(turn, chat_adapter)

    messages = turn.to_chat_messages(chat_adapter.flatten_response_items)
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})
    collaboration_mode = _extract_collaboration_mode(messages, instructions)
    log_debug(
        "RESPONSES_REQUEST_DIAGNOSTICS",
        _responses_request_diagnostics(
            data,
            messages,
            model,
            model_type,
            upstream_name,
        ),
    )
    return _ResponsesTurnContext(
        state_machine=state_machine,
        turn=turn,
        model_type=model_type,
        upstream_name=upstream_name,
        base_url=base_url,
        api_key=api_key,
        upstream_model=upstream_model,
        is_deepseek=is_deepseek,
        chat_adapter=chat_adapter,
        tool_type_map=tool_type_map,
        inject_mcp=inject_mcp,
        messages=messages,
        collaboration_mode=collaboration_mode,
    )


def _build_responses_chat_payload(
    data: dict[str, Any],
    context: _ResponsesTurnContext,
    tools_raw: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        k: v for k, v in data.items()
        if k not in ("messages", "tools", "input", "instructions",
                     "previous_response_id", "stream", "store", "include")
    }
    payload["model"] = context.upstream_model
    payload["messages"] = context.messages
    payload["stream"] = False  # Always non-streaming internally
    if tools_raw and not context.inject_mcp:
        chat_tools = (
            context.chat_adapter.responses_tools_to_chat(tools_raw)
            if context.is_deepseek
            else [convert_responses_tool_to_chat(t) for t in tools_raw]
        )
        if chat_tools:
            payload["tools"] = chat_tools
    payload = context.chat_adapter.filter_request_payload(payload)
    return _apply_provider_defaults(payload, context.model_type)


def _responses_provider_result_from_llm(
    context: _ResponsesTurnContext,
    llm_response: dict[str, Any],
    parse_result: Any,
    retry_count: int,
) -> _ResponsesProviderResult:
    choice = llm_response.get("choices", [{}])[0]
    response_message = choice.get("message", {})
    output_items, output_text, native_tool_calls = (
        context.chat_adapter.chat_response_to_output_items(
            response_message,
            context.tool_type_map,
        )
    )

    usage = _responses_usage_from_provider(llm_response.get("usage", {}))
    tool_calls_list = []
    if context.inject_mcp and parse_result and parse_result.success:
        output_items, tool_calls_list = (
            _mirothinker_adapter.to_responses_tool_outputs(
                parse_result,
            )
        )
    elif native_tool_calls:
        tool_calls_list = native_tool_calls

    return _ResponsesProviderResult(
        llm_response=llm_response,
        parse_result=parse_result,
        retry_count=retry_count,
        response_message=response_message,
        output_items=output_items,
        output_text=output_text,
        tool_calls_list=tool_calls_list,
        usage=usage,
    )


def _retry_plan_mode_recoveries(
    context: _ResponsesTurnContext,
    payload: dict[str, Any],
    result: _ResponsesProviderResult,
    model: str,
) -> _ResponsesProviderResult:
    plan_mode_mutation = _plan_mode_mutation_violation(
        context.collaboration_mode,
        result.output_items,
    )
    if plan_mode_mutation:
        log_debug("PLAN_MODE_PROPOSED_PLAN_RETRY", {
            "model": model,
            "upstream": context.upstream_name,
            "violation": plan_mode_mutation,
            "response_preview": (result.response_message.get("content") or "")[:240],
        })
        _append_plan_mode_proposed_plan_feedback(
            payload,
            result.response_message,
        )
        result = _responses_provider_result_from_llm(
            context,
            make_llm_request(payload, context.base_url, context.api_key),
            None,
            result.retry_count,
        )

    _validate_plan_mode_output_items(
        context.collaboration_mode,
        result.output_items,
    )

    if _plan_mode_should_retry_with_request_user_input(
        context.collaboration_mode,
        result.output_items,
        result.response_message,
    ):
        log_debug("PLAN_MODE_REQUEST_USER_INPUT_RETRY", {
            "model": model,
            "upstream": context.upstream_name,
            "response_preview": result.response_message.get("content", "")[:240],
        })
        _append_plan_mode_retry_feedback(payload, result.response_message)
        result = _responses_provider_result_from_llm(
            context,
            make_llm_request(payload, context.base_url, context.api_key),
            None,
            result.retry_count,
        )
        _validate_plan_mode_output_items(
            context.collaboration_mode,
            result.output_items,
        )

    return result


def _commit_and_build_responses_body(
    context: _ResponsesTurnContext,
    result: _ResponsesProviderResult,
    model: str,
) -> dict[str, Any]:
    has_tool_output = bool(result.tool_calls_list)
    output_text = None if has_tool_output else result.output_text
    provider_state_updates = (
        {"deepseek": context.chat_adapter.dump_provider_state()}
        if context.is_deepseek
        else None
    )
    log_debug(
        "RESPONSES_RESPONSE_DIAGNOSTICS",
        _responses_response_diagnostics(
            result.llm_response,
            result.output_items,
        ),
    )
    context.state_machine.commit_response(
        context.turn,
        result.output_items,
        provider_state_updates,
    )

    response = {
        "id": context.turn.response_id,
        "object": "response",
        "created": result.llm_response.get("created", int(time.time())),
        "model": model,
        "output": result.output_items,
        "output_text": output_text,
        "usage": result.usage,
        "status": "completed",
    }
    if result.retry_count > 0:
        response["_metadata"] = {
            "rollback_attempts": result.retry_count,
            "rollback_success": bool(
                context.inject_mcp
                and result.parse_result
                and result.parse_result.success
            ),
        }
    if result.tool_calls_list:
        response["tool_calls"] = result.tool_calls_list

    log_debug("CLIENT_RESPONSE /v1/responses", {
        "status": "success",
        "model": model,
        "model_type": context.model_type,
        "has_tool_calls": bool(result.tool_calls_list),
        "tool_calls_count": len(result.tool_calls_list),
        "rollback_attempts": result.retry_count if result.retry_count else None,
        "session_items": len(context.turn.session.items),
        "usage": result.usage,
    })
    return response


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
        instructions = data.get("instructions", "")
        tools_raw = data.get("tools", [])
        normalized_tools = _normalize_responses_tools(tools_raw)
        client_requested_stream = data.get("stream", False)

        # Convert tools to Chat format for internal use
        tools = normalized_tools

        model_type, upstream_name, base_url, api_key, upstream_model = _resolve(
            model,
            data,
            None,
        )
        if (
            model_type == "responses_passthrough"
            and not _is_official_deepseek_base_url(base_url)
        ):
            return _handle_responses_passthrough(
                data,
                model,
                model_type,
                upstream_name,
                base_url,
                api_key,
                upstream_model,
                tools_raw,
                normalized_tools,
                client_requested_stream,
            )

        context = _prepare_router_owned_responses_context(
            data,
            model,
            instructions,
            tools_raw,
        )
        payload = _build_responses_chat_payload(data, context, tools_raw)

        # ── Run LLM ──
        llm_response, parse_result, retry_count, _response_text = (
            _run_llm_with_rollback(
                payload,
                context.base_url,
                context.api_key,
                tools,
                context.inject_mcp,
            )
        )
        result = _responses_provider_result_from_llm(
            context,
            llm_response,
            parse_result,
            retry_count,
        )
        result = _retry_plan_mode_recoveries(
            context,
            payload,
            result,
            model,
        )
        response = _commit_and_build_responses_body(context, result, model)

        if client_requested_stream:
            return _build_sse_response(
                context.turn.response_id,
                result.output_items,
                result.usage,
            )

        return jsonify(response), 200

    except ResponsesStateError as e:
        log_debug("CLIENT_RESPONSE /v1/responses", {
            "status": "state_error",
            "error_code": e.code,
            "error": e.message,
        })
        return jsonify(e.to_error_dict()), e.status_code

    except LLMRequestError as e:
        payload_for_error = locals().get("payload")
        context_for_error = locals().get("context")
        is_deepseek_for_error = bool(
            getattr(context_for_error, "is_deepseek", False)
            or locals().get("is_deepseek", False)
        )
        error_body, status_code = _llm_request_error_body(
            e,
            payload=payload_for_error,
            is_deepseek=is_deepseek_for_error,
        )
        log_debug("CLIENT_RESPONSE /v1/responses", {
            "status": "provider_error",
            "error_code": error_body["error"].get("code"),
            "error": error_body["error"].get("message"),
            "provider_status": e.status_code,
        })
        return jsonify(error_body), status_code

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
        model_type, upstream_name, base_url, api_key, upstream_model = _resolve(model)
        tools = data.get("tools", [])
        messages = [m.copy() for m in data.get("messages", [])]

        payload = {
            k: v for k, v in data.items()
            if k not in ("messages", "tools")
        }
        payload["model"] = upstream_model
        payload["messages"] = messages
        payload["stream"] = False  # Always non-streaming internally
        chat_adapter = _chat_adapter_for(upstream_name, base_url)
        payload = chat_adapter.filter_request_payload(payload)
        payload = _apply_provider_defaults(payload, model_type)

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

    except LLMRequestError as e:
        payload_for_error = locals().get("payload")
        upstream_name_for_error = str(locals().get("upstream_name", ""))
        base_url_for_error = str(locals().get("base_url", ""))
        error_body, status_code = _llm_request_error_body(
            e,
            payload=payload_for_error,
            is_deepseek=_is_deepseek_upstream(
                upstream_name_for_error,
                base_url_for_error,
            ),
        )
        log_debug("CLIENT_RESPONSE /v1/chat/completions", {
            "status": "provider_error",
            "error_code": error_body["error"].get("code"),
            "error": error_body["error"].get("message"),
            "provider_status": e.status_code,
        })
        return jsonify(error_body), status_code

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
