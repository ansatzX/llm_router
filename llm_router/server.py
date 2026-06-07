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
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from flask import Flask, Response, jsonify, request, stream_with_context

from llm_router.chat_adapter_base import ChatCompletionAdapterBase
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
from llm_router.deepseek.anthropic_web_search import (
    DeepSeekAnthropicSearchExecution,
    DeepSeekAnthropicWebSearchBridge,
    DeepSeekAnthropicWebSearchExecutor,
    _anthropic_request_options_from_responses_payload,
    _queries_from_internal_tool_arguments,
)
from llm_router.llm_client import (
    LLMRequestError,
    ResponsesPassthroughError,
    _get_env_float,
    check_backend_health,
    list_models,
    make_llm_request,
    make_llm_stream_request,
    make_responses_request,
)
from llm_router.mirothinker import MiroThinkerMCPAdapter
from llm_router.openai_chat import OpenAIChatAdapter
from llm_router.provider_errors import _llm_request_error_body
from llm_router.reasoning_summary import reasoning_summary_text
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
from llm_router.xiaomi import XiaomiChatAdapter

logger = logging.getLogger(__name__)

_config: RouterConfig | None = None
_sessions: SessionStore | None = None
_deepseek_adapter = DeepSeekChatAdapter()
_mirothinker_adapter = MiroThinkerMCPAdapter()

app = Flask(__name__)

_MEMORY_MODEL_ALIASES = {
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
    provider_state_key: str | None
    chat_adapter: ChatCompletionAdapterBase
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
    provider_state_updates: dict[str, Any] | None = None


@dataclass
class _BuiltinWebSearchResult:
    output: str | None
    usage: dict[str, Any]


_XIAOMI_DO_WEB_SEARCH_TOOL_NAME = "do_web_search"
_XIAOMI_BUILTIN_WEB_SEARCH_MODEL = "mimo-v2-omni"
_XIAOMI_BUILTIN_TOOL_MAX_ROUNDS = 5
_XIAOMI_BUILTIN_TOOL_LIMIT_MESSAGE = (
    "已经多次搜索了，是否继续？如果确实必须继续搜索，请再次调用 do_web_search；"
    "否则请基于已有搜索结果回答用户。"
)
_XIAOMI_BUILTIN_TOOL_LIMIT_SUMMARY = "正在多次搜索，提醒用户"
_DEEPSEEK_WEB_SEARCH_TOOL_NAME = "__router_deepseek_web_search"
_DEEPSEEK_INTERNAL_WEB_SEARCH_MAX_ROUNDS = 5


def create_app(config_path: str | None = None, enable_dashboard: bool = False) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_path: Path to router.toml. Searches default locations if None.
        enable_dashboard: Register the /dashboard web UI blueprint.
    """
    global _config, _sessions

    _config = RouterConfig.from_toml(config_path) if config_path else RouterConfig.find_and_load()

    _sessions = SessionStore(ttl_seconds=_config.session_ttl_seconds)
    _deepseek_adapter.reset()

    if enable_dashboard:
        from llm_router.windows.dashboard import dashboard_bp
        app.register_blueprint(dashboard_bp)

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


def _is_xiaomi_upstream(upstream_name: str, base_url: str) -> bool:
    if upstream_name == "xiaomi":
        return True
    host = urlparse(base_url.rstrip("/")).netloc.lower()
    return host in {
        "api.xiaomimimo.com",
        "token-plan-cn.xiaomimimo.com",
        "token-plan-sgp.xiaomimimo.com",
        "token-plan-ams.xiaomimimo.com",
    }


def _is_official_deepseek_base_url(base_url: str) -> bool:
    return urlparse(base_url.rstrip("/")).netloc.lower() == "api.deepseek.com"


def _chat_adapter_for(upstream_name: str, base_url: str) -> ChatCompletionAdapterBase:
    if _is_deepseek_upstream(upstream_name, base_url):
        return DeepSeekChatAdapter()
    if _is_xiaomi_upstream(upstream_name, base_url):
        return XiaomiChatAdapter()
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


def _merge_reasoning_provider_states(
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


def _emit_sse(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


def _streaming_failure_event(
    response_id: str,
    message: str,
) -> str:
    return _emit_sse(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "status": "failed",
                "error": {"message": message},
            },
        },
    )


def _supports_live_upstream_streaming(
    context: _ResponsesTurnContext,
    tools_raw: list[dict[str, Any]],
) -> bool:
    """Return True when this turn can use real upstream streaming safely.

    Phase 1 scope:
    - router-owned `/v1/responses` only
    - no MCP-first rollback parser
    - no tool-call streaming assembly yet
    - no Plan-mode retry steering that requires hidden second attempts
    """
    if context.inject_mcp:
        return False
    if context.provider_state_key == "deepseek" and any(
        tool.get("type") == "web_search" for tool in tools_raw
    ):
        return False
    if context.provider_state_key == "xiaomi" and any(
        tool.get("type") == "web_search" for tool in tools_raw
    ):
        return False
    if tools_raw and context.provider_state_key not in {"deepseek", "xiaomi"}:
        return False
    return context.collaboration_mode != "Plan"


def _stream_tool_added_item(
    tool_name: str,
    call_id: str,
    tool_type_map: dict[str, Any],
) -> dict[str, Any]:
    tool_mapping = tool_type_map.get(tool_name)
    if tool_mapping == "custom":
        return {
            "type": "custom_tool_call",
            "id": call_id,
            "call_id": call_id,
            "name": tool_name,
            "input": "",
        }
    if (
        isinstance(tool_mapping, dict)
        and tool_mapping.get("type") == "namespace_function"
    ):
        return {
            "type": "function_call",
            "id": call_id,
            "call_id": call_id,
            "namespace": tool_mapping.get("namespace", ""),
            "name": tool_mapping.get("name", tool_name),
            "arguments": "",
        }
    return {
        "type": "function_call",
        "id": call_id,
        "call_id": call_id,
        "name": tool_name,
        "arguments": "",
    }


def _streamed_custom_input_prefix(arguments: str) -> str | None:
    """Return the stable decoded prefix of a Chat-wrapped custom tool input."""
    marker = '"input"'
    marker_pos = arguments.find(marker)
    if marker_pos < 0:
        return None
    colon_pos = arguments.find(":", marker_pos + len(marker))
    if colon_pos < 0:
        return None
    quote_pos = arguments.find('"', colon_pos + 1)
    if quote_pos < 0:
        return None

    decoded: list[str] = []
    i = quote_pos + 1
    while i < len(arguments):
        char = arguments[i]
        if char == '"':
            return "".join(decoded)
        if char != "\\":
            decoded.append(char)
            i += 1
            continue
        if i + 1 >= len(arguments):
            return "".join(decoded)
        escaped = arguments[i + 1]
        if escaped == "n":
            decoded.append("\n")
            i += 2
        elif escaped == "r":
            decoded.append("\r")
            i += 2
        elif escaped == "t":
            decoded.append("\t")
            i += 2
        elif escaped == "b":
            decoded.append("\b")
            i += 2
        elif escaped == "f":
            decoded.append("\f")
            i += 2
        elif escaped in {'"', "\\", "/"}:
            decoded.append(escaped)
            i += 2
        elif escaped == "u":
            if i + 6 > len(arguments):
                return "".join(decoded)
            hex_value = arguments[i + 2:i + 6]
            try:
                decoded.append(chr(int(hex_value, 16)))
            except ValueError:
                return "".join(decoded)
            i += 6
        else:
            decoded.append(escaped)
            i += 2
    return "".join(decoded)


def _custom_tool_input_delta(
    context: _ResponsesTurnContext,
    arguments: str,
    emitted_input: str,
) -> tuple[str, str]:
    custom_input = _streamed_custom_input_prefix(arguments)
    if custom_input is None:
        custom_input = context.chat_adapter.chat_tool_arguments_to_custom_input(arguments)
        looks_wrapped = arguments.lstrip().startswith("{") and '"input"' in arguments
        if custom_input == arguments and looks_wrapped:
            return "", emitted_input

    if custom_input.startswith(emitted_input):
        return custom_input[len(emitted_input):], custom_input
    if not emitted_input:
        return custom_input, custom_input
    return "", emitted_input


def _live_stream_router_owned_responses(
    context: _ResponsesTurnContext,
    payload: dict[str, Any],
    model: str,
) -> Response:
    """Proxy upstream Chat streaming and emit Responses SSE in real time.

    Commit semantics stay unchanged: session state is committed only after the
    upstream stream completes and the final assistant message is reconstructed.
    """
    payload = dict(payload)
    payload["stream"] = True

    response_id = context.turn.response_id
    created_ts = int(time.time())

    def _sync_final_output_item_ids(
        output_items: list[dict[str, Any]],
        reasoning_id: str,
        message_id: str,
        has_reasoning: bool,
        has_message: bool,
    ) -> None:
        """Keep streamed added/delta items and final done/completed items identical."""
        for item in output_items:
            item_type = item.get("type")
            if item_type == "reasoning" and has_reasoning:
                item["id"] = reasoning_id
            elif item_type == "message" and has_message:
                item["id"] = message_id

    def _generate() -> Any:
        reasoning_item_id = f"rsn_{uuid.uuid4().hex[:8]}"
        message_item_id = f"msg_{uuid.uuid4().hex[:8]}"

        reasoning_started = False
        reasoning_part_started = False
        message_started = False

        reasoning_parts: list[str] = []
        message_parts: list[str] = []
        tool_states: dict[int, dict[str, Any]] = {}
        tool_order: list[int] = []
        tool_seen_order: list[int] = []
        tool_index_by_id: dict[str, int] = {}
        tool_next_anon_index = -1
        usage_raw: dict[str, Any] = {}
        finish_reason: str | None = None

        yield _emit_sse(
            "response.created",
            {
                "type": "response.created",
                "response": {"id": response_id},
            },
        )

        try:
            for chunk in make_llm_stream_request(
                payload,
                context.base_url,
                context.api_key,
            ):
                choices = chunk.get("choices") or []
                if choices:
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice.get("delta") or {}

                    reasoning_delta = delta.get("reasoning_content")
                    if isinstance(reasoning_delta, str) and reasoning_delta:
                        if message_started or tool_seen_order:
                            raise LLMRequestError(
                                "Late streamed reasoning after output items is unsupported.",
                            )
                        if not reasoning_started:
                            reasoning_started = True
                            yield _emit_sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": 0,
                                    "item": {
                                        "type": "reasoning",
                                        "id": reasoning_item_id,
                                        "summary": [
                                            {"type": "summary_text", "text": ""},
                                        ],
                                        "content": [
                                            {"type": "reasoning_text", "text": ""},
                                        ],
                                    },
                                },
                            )
                        if not reasoning_part_started:
                            reasoning_part_started = True
                            yield _emit_sse(
                                "response.reasoning_summary_part.added",
                                {
                                    "type": "response.reasoning_summary_part.added",
                                    "output_index": 0,
                                    "item_id": reasoning_item_id,
                                    "summary_index": 0,
                                    "part": {"type": "summary_text"},
                                },
                            )

                        reasoning_parts.append(reasoning_delta)
                        yield _emit_sse(
                            "response.reasoning_text.delta",
                            {
                                "type": "response.reasoning_text.delta",
                                "output_index": 0,
                                "item_id": reasoning_item_id,
                                "content_index": 0,
                                "delta": reasoning_delta,
                            },
                        )

                    text_delta = delta.get("content")
                    if isinstance(text_delta, str) and text_delta:
                        # Allow text after tool_calls for providers that stream both
                        message_index = 1 if reasoning_started else 0
                        if not message_started:
                            message_started = True
                            yield _emit_sse(
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": message_index,
                                    "item": {
                                        "type": "message",
                                        "id": message_item_id,
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": ""}],
                                    },
                                },
                            )
                        message_parts.append(text_delta)
                        yield _emit_sse(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "output_index": message_index,
                                "item_id": message_item_id,
                                "delta": text_delta,
                            },
                        )

                    tool_call_deltas = delta.get("tool_calls")
                    if isinstance(tool_call_deltas, list) and tool_call_deltas:
                        # Allow tool_calls after text for providers that stream both
                        _message_emitted = message_started
                        for tool_call_delta in tool_call_deltas:
                            if not isinstance(tool_call_delta, dict):
                                continue
                            call_id = tool_call_delta.get("id")
                            index = tool_call_delta.get("index")
                            if not isinstance(index, int):
                                if isinstance(call_id, str) and call_id in tool_index_by_id:
                                    index = tool_index_by_id[call_id]
                                else:
                                    index = tool_next_anon_index
                                    tool_next_anon_index -= 1
                            if (
                                isinstance(call_id, str)
                                and call_id
                                and call_id in tool_index_by_id
                            ):
                                index = tool_index_by_id[call_id]
                            if isinstance(call_id, str) and call_id:
                                tool_index_by_id[call_id] = index
                            state = tool_states.setdefault(
                                index,
                                {
                                    "id": None,
                                    "name": None,
                                    "arguments_parts": [],
                                    "emitted_argument_parts": 0,
                                    "custom_input_emitted": "",
                                    "output_index": None,
                                    "added": False,
                                },
                            )
                            if index not in tool_seen_order:
                                tool_seen_order.append(index)
                            if isinstance(call_id, str) and call_id:
                                state["id"] = call_id

                            function_delta = tool_call_delta.get("function")
                            if isinstance(function_delta, dict):
                                tool_name = function_delta.get("name")
                                if isinstance(tool_name, str) and tool_name:
                                    state["name"] = tool_name
                                arguments_delta = function_delta.get("arguments")
                                if isinstance(arguments_delta, str):
                                    state["arguments_parts"].append(arguments_delta)

                            # Account for message item already occupying an output slot
                            _base_index = 1 if reasoning_started else 0
                            if _message_emitted:
                                _base_index += 1
                            if (
                                not state["added"]
                                and state.get("name")
                                and state.get("id")
                            ):
                                state["output_index"] = (
                                    _base_index + len(tool_order)
                                )
                                added_item = _stream_tool_added_item(
                                    state["name"],
                                    state["id"],
                                    context.tool_type_map,
                                )
                                tool_order.append(index)
                                state["added"] = True
                                yield _emit_sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": state["output_index"],
                                        "item": added_item,
                                    },
                                )

                            if not state["added"]:
                                continue

                            arguments_parts = state.get("arguments_parts", [])
                            emitted_parts = int(state.get("emitted_argument_parts", 0))
                            if emitted_parts >= len(arguments_parts):
                                continue
                            tool_mapping = context.tool_type_map.get(
                                state.get("name"),
                                "function",
                            )
                            if tool_mapping == "custom":
                                arguments = "".join(arguments_parts)
                                custom_delta, emitted_input = _custom_tool_input_delta(
                                    context,
                                    arguments,
                                    str(state.get("custom_input_emitted", "")),
                                )
                                if custom_delta:
                                    yield _emit_sse(
                                        "response.custom_tool_call_input.delta",
                                        {
                                            "type": "response.custom_tool_call_input.delta",
                                            "output_index": state["output_index"],
                                            "item_id": state["id"],
                                            "call_id": state["id"],
                                            "delta": custom_delta,
                                        },
                                    )
                                    state["custom_input_emitted"] = emitted_input
                            else:
                                for arguments_delta in arguments_parts[emitted_parts:]:
                                    if not arguments_delta:
                                        continue
                                    yield _emit_sse(
                                        "response.function_call_arguments.delta",
                                        {
                                            "type": "response.function_call_arguments.delta",
                                            "output_index": state["output_index"],
                                            "item_id": state["id"],
                                            "call_id": state["id"],
                                            "delta": arguments_delta,
                                        },
                                    )
                            state["emitted_argument_parts"] = len(arguments_parts)

                    finish_reason = choice.get("finish_reason") or finish_reason

                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage_raw = chunk_usage

            response_message: dict[str, Any] = {}
            if message_parts:
                response_message["content"] = "".join(message_parts)
            else:
                response_message["content"] = ""
            if reasoning_parts:
                response_message["reasoning_content"] = "".join(reasoning_parts)
                if tool_seen_order:
                    response_message["reasoning_summary"] = ""
            if tool_seen_order:
                for index in tool_seen_order:
                    state = tool_states[index]
                    if not state.get("id"):
                        state["id"] = f"call_{index}_{uuid.uuid4().hex[:6]}"
                    if not state["added"]:
                        if not state.get("name"):
                            state["name"] = "unknown_tool"
                        _post_base = 1 if reasoning_started else 0
                        if message_started:
                            _post_base += 1
                        state["output_index"] = _post_base + len(tool_order)
                        added_item = _stream_tool_added_item(
                            state["name"],
                            state["id"],
                            context.tool_type_map,
                        )
                        state["added"] = True
                        yield _emit_sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": state["output_index"],
                                "item": added_item,
                            },
                        )
                        tool_order.append(index)

                native_tool_calls: list[dict[str, Any]] = []
                for index in tool_order:
                    state = tool_states[index]
                    arguments = "".join(state.get("arguments_parts", []))
                    tool_name = state.get("name") or "unknown_tool"
                    call_id = state["id"]

                    if context.tool_type_map.get(tool_name) == "custom":
                        final_custom_delta, emitted_input = _custom_tool_input_delta(
                            context,
                            arguments,
                            str(state.get("custom_input_emitted", "")),
                        )
                        if final_custom_delta:
                            yield _emit_sse(
                                "response.custom_tool_call_input.delta",
                                {
                                    "type": "response.custom_tool_call_input.delta",
                                    "output_index": state["output_index"],
                                    "item_id": call_id,
                                    "call_id": call_id,
                                    "delta": final_custom_delta,
                                },
                            )
                        state["custom_input_emitted"] = emitted_input

                    native_tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": arguments,
                            },
                        },
                    )
                response_message["tool_calls"] = native_tool_calls

            llm_response = {
                "created": created_ts,
                "choices": [
                    {
                        "message": response_message,
                        "finish_reason": finish_reason or "stop",
                    },
                ],
                "usage": usage_raw,
            }

            result = _responses_provider_result_from_llm(
                context,
                llm_response,
                parse_result=None,
                retry_count=0,
            )
            _sync_final_output_item_ids(
                result.output_items,
                reasoning_item_id,
                message_item_id,
                reasoning_started,
                message_started,
            )
            response_body = _commit_and_build_responses_body(context, result, model)
            reasoning_summary = _first_reasoning_summary_text(result.output_items)
            if reasoning_started and reasoning_summary:
                yield _emit_sse(
                    "response.reasoning_summary_text.delta",
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": 0,
                        "item_id": reasoning_item_id,
                        "summary_index": 0,
                        "delta": reasoning_summary,
                    },
                )

            for idx, item in enumerate(result.output_items):
                item_done_event = {
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": item,
                }
                yield _emit_sse("response.output_item.done", item_done_event)

            yield _emit_sse(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": response_body,
                },
            )
        except Exception as exc:
            logger.exception("Error in live /v1/responses stream")
            message = getattr(exc, "message", str(exc))
            yield _streaming_failure_event(response_id, message)

    return Response(
        stream_with_context(_generate()),
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


def _load_reasoning_provider_state(
    turn: ResponsesTurn,
    chat_adapter: ChatCompletionAdapterBase,
    provider_state_key: str,
) -> None:
    call_ids = _tool_call_ids_from_response_items(
        [*turn.session.items, *turn.input_items],
    )
    recovered_provider_state = _sessions.provider_state_for_call_ids(
        provider_state_key,
        call_ids,
    )
    session_provider_state = turn.session.provider_state.get(provider_state_key)
    chat_adapter.load_provider_state(
        _merge_reasoning_provider_states(
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
        log_debug("REASONING_PROVIDER_STATE_RECOVERY", {
            "provider": provider_state_key,
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
    is_xiaomi = _is_xiaomi_upstream(upstream_name, base_url)
    provider_state_key = (
        "deepseek" if is_deepseek else "xiaomi" if is_xiaomi else None
    )
    chat_adapter = _chat_adapter_for(upstream_name, base_url)
    tool_type_map = chat_adapter.tool_type_map(tools_raw)
    inject_mcp = (model_type == "mcp_first")
    if provider_state_key:
        _load_reasoning_provider_state(turn, chat_adapter, provider_state_key)

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
        provider_state_key=provider_state_key,
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
            if context.provider_state_key in {"deepseek", "xiaomi"}
            else [convert_responses_tool_to_chat(t) for t in tools_raw]
        )
        if chat_tools:
            payload["tools"] = chat_tools
    payload = context.chat_adapter.filter_request_payload(payload)
    return _apply_provider_defaults(payload, context.model_type)


def _is_active_web_search_tool(tool: Any) -> bool:
    return (
        isinstance(tool, dict)
        and tool.get("type") == "web_search"
        and tool.get("external_web_access") is not False
    )


def _split_builtin_web_search_tools(
    tools_raw: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    web_search_tools: list[dict[str, Any]] = []
    remaining_tools: list[dict[str, Any]] = []
    for tool in tools_raw:
        if _is_active_web_search_tool(tool):
            web_search_tools.append(tool)
        else:
            remaining_tools.append(tool)
    return web_search_tools, remaining_tools


def _deepseek_active_web_search_tool(
    context: _ResponsesTurnContext,
    tools_raw: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if context.provider_state_key != "deepseek":
        return None
    if context.model_type != "responses_chat":
        return None
    if not _is_official_deepseek_base_url(context.base_url):
        return None
    web_search_tools, _remaining_tools = _split_builtin_web_search_tools(tools_raw)
    return web_search_tools[0] if web_search_tools else None


def _deepseek_web_search_internal_tool(
    web_search_tool: dict[str, Any],
) -> dict[str, Any]:
    """Expose DeepSeek hosted web_search as a model-chosen internal function."""
    description = (
        "Search the live web when current or external information is needed. "
        "Call this tool only when the user request requires web results."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Focused web search query.",
            },
            "queries": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Focused web search query.",
                },
                "description": "Focused web search queries.",
            },
        },
        "additionalProperties": False,
    }
    for key in (
        "search_context_size",
        "filters",
        "location",
        "user_location",
        "force_search",
        "forced_search",
        "max_keyword",
        "limit",
    ):
        if key in web_search_tool:
            parameters.setdefault("x-router-deepseek-web-search-defaults", {})[key] = (
                web_search_tool[key]
            )
    return {
        "type": "function",
        "name": _DEEPSEEK_WEB_SEARCH_TOOL_NAME,
        "description": description,
        "parameters": parameters,
    }


def _replace_deepseek_web_search_with_internal_tool(
    context: _ResponsesTurnContext,
    tools_raw: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if _deepseek_active_web_search_tool(context, tools_raw) is None:
        return None, tools_raw
    web_search_tools, remaining_tools = _split_builtin_web_search_tools(tools_raw)
    internal_tool = _deepseek_web_search_internal_tool(web_search_tools[0])
    return web_search_tools[0], [*remaining_tools, internal_tool]


def _find_deepseek_web_search_calls(
    native_tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for tool_call in native_tool_calls:
        function = tool_call.get("function") or {}
        if function.get("name") != _DEEPSEEK_WEB_SEARCH_TOOL_NAME:
            continue
        queries = _queries_from_internal_tool_arguments(function.get("arguments", ""))
        calls.append({
            "id": str(tool_call.get("id") or ""),
            "queries": queries,
            "query": queries[0] if queries else "",
        })
    return calls


def _deepseek_web_search_context_message(
    call: dict[str, Any],
    execution: DeepSeekAnthropicSearchExecution,
) -> str:
    parts = [
        "Router web search results are available for this turn.",
    ]
    query = str(call.get("query") or "").strip()
    if query:
        parts.append(f"Query: {query}")
    if execution.text:
        parts.extend(["Result:", execution.text])
    if execution.searches:
        parts.append("Sources:")
        for index, search in enumerate(execution.searches, start=1):
            search_query = str(search.get("query") or "").strip()
            if search_query:
                parts.append(f"{index}. {search_query}")
            search_text = search.get("text")
            if isinstance(search_text, str) and search_text:
                parts.append(f"   {search_text}")
            else:
                for result in (search.get("results") or [])[:3]:
                    if not isinstance(result, dict):
                        continue
                    title = result.get("title") or result.get("url") or "source"
                    url = result.get("url") or ""
                    line = f"   - {title}"
                    if url:
                        line = f"{line}: {url}"
                    parts.append(line)
    return "\n".join(parts)


def _deepseek_web_search_output_item(call: dict[str, Any]) -> dict[str, Any]:
    query = str(call.get("query") or "").strip()
    action: dict[str, Any]
    if query:
        action = {"type": "search", "query": query}
    else:
        queries = call.get("queries") or []
        if isinstance(queries, list) and queries:
            action = {
                "type": "search",
                "query": "\n".join(
                    str(query).strip()
                    for query in queries
                    if isinstance(query, str) and str(query).strip()
                ),
            }
        else:
            action = {"type": "other"}
    return {
        "type": "web_search_call",
        "id": str(call.get("id") or ""),
        "status": "completed",
        "action": action,
    }


def _xiaomi_do_web_search_tool(web_search_tool: dict[str, Any]) -> dict[str, Any]:
    """Expose Xiaomi hosted search as a model-chosen internal function."""
    description = (
        "Search the live web when current or external information is needed. "
        "Call this tool only when the user request requires web results. "
        "The router will execute Xiaomi MiMo web_search and return text or null."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Focused web search query.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    for key in ("max_keyword", "limit", "force_search", "forced_search", "user_location"):
        if key in web_search_tool:
            parameters.setdefault("x-xiaomi-web-search-defaults", {})[key] = (
                web_search_tool[key]
            )
    return {
        "type": "function",
        "name": _XIAOMI_DO_WEB_SEARCH_TOOL_NAME,
        "description": description,
        "parameters": parameters,
    }


def _replace_xiaomi_web_search_with_internal_tool(
    context: _ResponsesTurnContext,
    tools_raw: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if context.provider_state_key != "xiaomi":
        return None, tools_raw
    web_search_tools, remaining_tools = _split_builtin_web_search_tools(tools_raw)
    if not web_search_tools:
        return None, tools_raw
    internal_tool = _xiaomi_do_web_search_tool(web_search_tools[0])
    return web_search_tools[0], [*remaining_tools, internal_tool]


def _parse_do_web_search_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip()
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        return ""
    if isinstance(parsed, dict):
        query = parsed.get("query") or parsed.get("input") or parsed.get("q")
        if isinstance(query, str):
            return query.strip()
    return ""


def _format_search_annotations(annotations: list[Any]) -> str:
    lines: list[str] = []
    for index, annotation in enumerate(annotations, start=1):
        if not isinstance(annotation, dict):
            continue
        title = annotation.get("title") or annotation.get("site_name") or "source"
        url = annotation.get("url") or ""
        summary = annotation.get("summary") or ""
        source = f"{index}. {title}"
        if url:
            source = f"{source}: {url}"
        lines.append(source)
        if summary:
            lines.append(f"   {summary}")
    return "\n".join(lines)


def _build_search_context_message(
    query: str,
    search_message: dict[str, Any],
) -> dict[str, str]:
    content = search_message.get("content") or ""
    annotations = search_message.get("annotations")
    if not isinstance(annotations, list):
        annotations = []
    sources = _format_search_annotations(annotations)
    parts = [
        "Router web search results are available for this turn.",
        f"Query: {query}",
    ]
    if content:
        parts.extend(["Result:", str(content)])
    if sources:
        parts.extend(["Sources:", sources])
    return {
        "role": "developer",
        "content": "\n".join(parts),
    }


def _run_xiaomi_builtin_web_search(
    data: dict[str, Any],
    context: _ResponsesTurnContext,
    web_search_tool: dict[str, Any],
    query: str,
) -> _BuiltinWebSearchResult:
    search_tools = context.chat_adapter.responses_tools_to_chat([web_search_tool])
    payload = {
        "model": _XIAOMI_BUILTIN_WEB_SEARCH_MODEL,
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "tools": search_tools,
        "tool_choice": "auto",
        "max_completion_tokens": data.get("max_completion_tokens", 1024),
        "thinking": {"type": "disabled"},
    }
    for key in (
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
    ):
        if key in data:
            payload[key] = data[key]
    payload = context.chat_adapter.filter_request_payload(payload)
    log_debug("XIAOMI_BUILTIN_WEB_SEARCH_REQUEST", {
        "model": payload["model"],
        "query_preview": query[:240],
        "tool_count": len(payload.get("tools", []) or []),
    })
    response = make_llm_request(payload, context.base_url, context.api_key)
    message = ((response.get("choices") or [{}])[0] or {}).get("message") or {}
    output = _build_search_context_message(query, message)["content"]
    return _BuiltinWebSearchResult(
        output=output if output else None,
        usage=_responses_usage_from_provider(response.get("usage", {})),
    )


def _run_xiaomi_builtin_web_search_safely(
    data: dict[str, Any],
    context: _ResponsesTurnContext,
    web_search_tool: dict[str, Any] | None,
    query: str,
) -> _BuiltinWebSearchResult:
    empty_usage = _responses_usage_from_provider({})
    if not web_search_tool:
        log_debug("XIAOMI_BUILTIN_WEB_SEARCH_SKIPPED", {
            "reason": "no_xiaomi_web_search_tool",
            "query_preview": query[:240],
        })
        return _BuiltinWebSearchResult(output=None, usage=empty_usage)
    if not query:
        log_debug("XIAOMI_BUILTIN_WEB_SEARCH_SKIPPED", {
            "reason": "empty_query",
        })
        return _BuiltinWebSearchResult(output=None, usage=empty_usage)
    try:
        return _run_xiaomi_builtin_web_search(
            data,
            context,
            web_search_tool,
            query,
        )
    except LLMRequestError as exc:
        log_debug("XIAOMI_BUILTIN_WEB_SEARCH_FAILED", {
            "query_preview": query[:240],
            "provider_status": exc.status_code,
            "message": exc.message,
            "body": exc.body,
        })
        return _BuiltinWebSearchResult(output=None, usage=empty_usage)


def _run_deepseek_internal_tool_loop(
    data: dict[str, Any],
    context: _ResponsesTurnContext,
    payload: dict[str, Any],
    web_search_tool: dict[str, Any] | None,
    first_result: _ResponsesProviderResult,
) -> _ResponsesProviderResult:
    result = first_result
    if not web_search_tool:
        return result

    total_tool_usage = _responses_usage_from_provider({})
    search_output_items: list[dict[str, Any]] = []
    search_rounds = 0

    while True:
        calls = _find_deepseek_web_search_calls(
            result.response_message.get("tool_calls") or [],
        )
        if not calls:
            break
        if search_rounds >= _DEEPSEEK_INTERNAL_WEB_SEARCH_MAX_ROUNDS:
            log_debug("DEEPSEEK_INTERNAL_TOOL_LOOP_STOPPED", {
                "reason": "max_rounds_exhausted",
                "max_rounds": _DEEPSEEK_INTERNAL_WEB_SEARCH_MAX_ROUNDS,
                "queries": [call["query"][:240] for call in calls],
            })
            raise LLMRequestError(
                "DeepSeek hosted web search exhausted its internal tool budget.",
                status_code=502,
            )

        log_debug("DEEPSEEK_INTERNAL_WEB_SEARCH_TOOL_CALLS", {
            "round": search_rounds + 1,
            "call_count": len(calls),
            "queries": [call["query"][:240] for call in calls],
        })

        tool_outputs: list[dict[str, Any]] = []
        for call in calls:
            queries = call["queries"] or ([call["query"]] if call["query"] else [])
            if not queries:
                raise LLMRequestError(
                    "DeepSeek hosted web search call is missing query arguments.",
                    status_code=502,
                )
            executor = DeepSeekAnthropicWebSearchExecutor(
                context.base_url,
                context.api_key,
                context.upstream_model,
            )
            search_result = executor.execute(queries)
            total_tool_usage = _combine_responses_usage(
                total_tool_usage,
                search_result.usage,
            )
            tool_outputs.append({
                "call_id": call["id"],
                "output": _deepseek_web_search_context_message(call, search_result),
            })
            search_output_items.append(_deepseek_web_search_output_item(call))

        _append_internal_tool_exchange(
            payload,
            result.response_message,
            tool_outputs,
        )
        result = _responses_provider_result_from_llm(
            context,
            make_llm_request(payload, context.base_url, context.api_key),
            None,
            result.retry_count,
        )
        search_rounds += 1

    if search_output_items:
        result.output_items = [*search_output_items, *result.output_items]
    result.usage = _combine_responses_usage(total_tool_usage, result.usage)
    return result


def _combine_responses_usage(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input_tokens": int(first.get("input_tokens", 0)) + int(second.get("input_tokens", 0)),
        "input_tokens_details": {
            "cached_tokens": (
                int((first.get("input_tokens_details") or {}).get("cached_tokens", 0))
                + int((second.get("input_tokens_details") or {}).get("cached_tokens", 0))
            ),
        },
        "output_tokens": int(first.get("output_tokens", 0)) + int(second.get("output_tokens", 0)),
        "output_tokens_details": {
            "reasoning_tokens": (
                int((first.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
                + int((second.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
            ),
        },
        "total_tokens": int(first.get("total_tokens", 0)) + int(second.get("total_tokens", 0)),
    }


def _codex_will_stop_after_completed(response_body: dict[str, Any]) -> bool:
    if response_body.get("end_turn") is False:
        return False
    for item in response_body.get("output", []):
        if isinstance(item, dict) and item.get("type") in {
            "function_call",
            "custom_tool_call",
        }:
            return False
    return True


def _update_reasoning_summary_visibility(
    output_items: list[dict[str, Any]],
    *,
    will_stop: bool,
) -> None:
    for item in output_items:
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if not isinstance(summary, list) or not summary:
            continue
        first_part = summary[0]
        if not isinstance(first_part, dict):
            continue
        reasoning_content = ""
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "reasoning_text":
                reasoning_content += str(part.get("text") or "")
        if not reasoning_content:
            continue
        first_part["text"] = reasoning_summary_text(
            reasoning_content,
            will_stop=will_stop,
        )
        break


def _first_reasoning_summary_text(output_items: list[dict[str, Any]]) -> str:
    for item in output_items:
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if not isinstance(summary, list) or not summary:
            return ""
        first_part = summary[0]
        if not isinstance(first_part, dict):
            return ""
        return str(first_part.get("text") or "")
    return ""


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


def _positive_int_param(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _numeric_param(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _deepseek_anthropic_bridge_max_tokens(data: dict[str, Any]) -> int | None:
    for key in ("max_tokens", "max_output_tokens", "max_completion_tokens"):
        value = _positive_int_param(data.get(key))
        if value is not None:
            return value
    return None


def _responses_tool_calls_from_output_items(
    output_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item for item in output_items
        if item.get("type") in {"function_call", "custom_tool_call"}
    ]


def _responses_provider_result_from_deepseek_anthropic_bridge(
    bridge_result: Any,
) -> _ResponsesProviderResult:
    output_items = bridge_result.output_items
    output_text = bridge_result.output_text
    tool_calls_list = (
        getattr(bridge_result, "tool_calls_list", None)
        or _responses_tool_calls_from_output_items(output_items)
    )
    raw_response = bridge_result.raw_response if isinstance(bridge_result.raw_response, dict) else {}
    response_message = {
        "role": "assistant",
        "content": output_text if output_text is not None else None,
    }
    llm_response = {
        "created": raw_response.get("created", int(time.time())),
        "choices": [
            {
                "message": response_message,
                "finish_reason": raw_response.get("stop_reason", "stop"),
            }
        ],
        "usage": bridge_result.usage,
    }
    return _ResponsesProviderResult(
        llm_response=llm_response,
        parse_result=None,
        retry_count=0,
        response_message=response_message,
        output_items=output_items,
        output_text=output_text,
        tool_calls_list=tool_calls_list,
        usage=bridge_result.usage,
    )


def _run_deepseek_anthropic_web_search_bridge(
    data: dict[str, Any],
    context: _ResponsesTurnContext,
    tools_raw: list[dict[str, Any]],
) -> _ResponsesProviderResult:
    bridge = DeepSeekAnthropicWebSearchBridge(
        context.base_url,
        context.api_key,
        context.upstream_model,
    )
    bridge_result = bridge.run(
        messages=context.messages,
        tools_raw=tools_raw,
        max_tokens=_deepseek_anthropic_bridge_max_tokens(data),
        temperature=_numeric_param(data.get("temperature")),
        top_p=_numeric_param(data.get("top_p")),
        request_options=_anthropic_request_options_from_responses_payload(data),
    )
    result = _responses_provider_result_from_deepseek_anthropic_bridge(
        bridge_result,
    )
    provider_state = getattr(bridge_result, "provider_state", None)
    if context.provider_state_key and isinstance(provider_state, dict):
        result.provider_state_updates = {
            context.provider_state_key: provider_state,
        }
    return result


def _find_do_web_search_calls(
    native_tool_calls: list[dict[str, Any]],
) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    for tool_call in native_tool_calls:
        function = tool_call.get("function") or {}
        if function.get("name") != _XIAOMI_DO_WEB_SEARCH_TOOL_NAME:
            continue
        calls.append({
            "id": str(tool_call.get("id") or ""),
            "query": _parse_do_web_search_arguments(function.get("arguments", "")),
        })
    return calls


def _append_internal_tool_exchange(
    payload: dict[str, Any],
    assistant_message: dict[str, Any],
    tool_outputs: list[dict[str, Any]],
) -> None:
    payload.setdefault("messages", []).append(assistant_message)
    for tool_output in tool_outputs:
        payload["messages"].append({
            "role": "tool",
            "tool_call_id": tool_output["call_id"],
            "content": json.dumps(tool_output["output"], ensure_ascii=False),
        })


def _prepend_xiaomi_search_reasoning_summary(
    result: _ResponsesProviderResult,
) -> None:
    if any(
        item.get("type") == "reasoning"
        and item.get("summary") == [
            {
                "type": "summary_text",
                "text": _XIAOMI_BUILTIN_TOOL_LIMIT_SUMMARY,
            }
        ]
        for item in result.output_items
    ):
        return
    result.output_items.insert(0, {
        "type": "reasoning",
        "summary": [
            {
                "type": "summary_text",
                "text": _XIAOMI_BUILTIN_TOOL_LIMIT_SUMMARY,
            },
        ],
        "content": [],
    })


def _run_xiaomi_internal_tool_loop(
    data: dict[str, Any],
    context: _ResponsesTurnContext,
    payload: dict[str, Any],
    web_search_tool: dict[str, Any] | None,
    first_result: _ResponsesProviderResult,
) -> _ResponsesProviderResult:
    result = first_result
    total_tool_usage = _responses_usage_from_provider({})
    search_rounds_since_question = 0
    questioned_after_many_searches = False
    while True:
        calls = _find_do_web_search_calls(result.response_message.get("tool_calls") or [])
        if not calls:
            break
        if search_rounds_since_question >= _XIAOMI_BUILTIN_TOOL_MAX_ROUNDS:
            questioned_after_many_searches = True
            log_debug("XIAOMI_INTERNAL_TOOL_LOOP_STOPPED", {
                "reason": "questioned_model_after_max_search_rounds",
                "max_rounds": _XIAOMI_BUILTIN_TOOL_MAX_ROUNDS,
                "queries": [call["query"][:240] for call in calls],
            })
            _append_internal_tool_exchange(
                payload,
                result.response_message,
                [
                    {
                        "call_id": call["id"],
                        "output": _XIAOMI_BUILTIN_TOOL_LIMIT_MESSAGE,
                    }
                    for call in calls
                ],
            )
            result = _responses_provider_result_from_llm(
                context,
                make_llm_request(payload, context.base_url, context.api_key),
                None,
                result.retry_count,
            )
            search_rounds_since_question = 0
            continue

        log_debug("XIAOMI_INTERNAL_WEB_SEARCH_TOOL_CALLS", {
            "round": search_rounds_since_question + 1,
            "call_count": len(calls),
            "queries": [call["query"][:240] for call in calls],
        })
        tool_outputs: list[dict[str, Any]] = []
        for call in calls:
            search_result = _run_xiaomi_builtin_web_search_safely(
                data,
                context,
                web_search_tool,
                call["query"],
            )
            total_tool_usage = _combine_responses_usage(
                total_tool_usage,
                search_result.usage,
            )
            tool_outputs.append({
                "call_id": call["id"],
                "output": search_result.output,
            })
        _append_internal_tool_exchange(
            payload,
            result.response_message,
            tool_outputs,
        )
        result = _responses_provider_result_from_llm(
            context,
            make_llm_request(payload, context.base_url, context.api_key),
            None,
            result.retry_count,
        )
        search_rounds_since_question += 1

    if questioned_after_many_searches:
        _prepend_xiaomi_search_reasoning_summary(result)

    if result.response_message.get("tool_calls"):
        log_debug("XIAOMI_INTERNAL_TOOL_LOOP_STOPPED", {
            "reason": "tool_calls_remain_after_max_rounds",
            "tool_names": [
                ((tool_call.get("function") or {}).get("name"))
                for tool_call in result.response_message.get("tool_calls", [])
                if isinstance(tool_call, dict)
            ],
        })
    result.usage = _combine_responses_usage(total_tool_usage, result.usage)
    return result


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
    provider_state_updates = dict(result.provider_state_updates or {})
    if context.provider_state_key:
        provider_state_updates[context.provider_state_key] = (
            _merge_reasoning_provider_states(
                provider_state_updates.get(context.provider_state_key),
                context.chat_adapter.dump_provider_state(),
            )
        )
    if not provider_state_updates:
        provider_state_updates = None
    log_debug(
        "RESPONSES_RESPONSE_DIAGNOSTICS",
        _responses_response_diagnostics(
            result.llm_response,
            result.output_items,
        ),
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

    _update_reasoning_summary_visibility(
        result.output_items,
        will_stop=_codex_will_stop_after_completed(response),
    )
    context.state_machine.commit_response(
        context.turn,
        result.output_items,
        provider_state_updates,
    )

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
        if _deepseek_active_web_search_tool(context, tools_raw) is not None:
            result = _run_deepseek_anthropic_web_search_bridge(
                data,
                context,
                tools_raw,
            )
            response = _commit_and_build_responses_body(context, result, model)
            if client_requested_stream:
                return _build_sse_response(
                    context.turn.response_id,
                    result.output_items,
                    result.usage,
                )
            return jsonify(response), 200

        deepseek_web_search_tool, main_tools_raw = _replace_deepseek_web_search_with_internal_tool(
            context,
            tools_raw,
        )
        xiaomi_web_search_tool, main_tools_raw = _replace_xiaomi_web_search_with_internal_tool(
            context,
            main_tools_raw,
        )
        payload = _build_responses_chat_payload(data, context, main_tools_raw)

        if (
            client_requested_stream
            and _supports_live_upstream_streaming(context, tools_raw)
        ):
            return _live_stream_router_owned_responses(
                context,
                payload,
                model,
            )

        # ── Run LLM ──
        llm_response, parse_result, retry_count, _response_text = (
            _run_llm_with_rollback(
                payload,
                context.base_url,
                context.api_key,
                main_tools_raw,
                context.inject_mcp,
            )
        )
        result = _responses_provider_result_from_llm(
            context,
            llm_response,
            parse_result,
            retry_count,
        )
        result = _run_deepseek_internal_tool_loop(
            data,
            context,
            payload,
            deepseek_web_search_tool,
            result,
        )
        result = _run_xiaomi_internal_tool_loop(
            data,
            context,
            payload,
            xiaomi_web_search_tool,
            result,
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
            if k != "messages"
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
