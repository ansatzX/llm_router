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
import re
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
from flask import Flask, Response, jsonify, request, stream_with_context

from llm_router.config import RouterConfig
from llm_router.debug_log import log_debug
from llm_router.deepseek import DeepSeekChatAdapter
from llm_router.llm_client import (
    ResponsesPassthroughUnsupportedError,
    _get_env_float,
    check_backend_health,
    list_models,
    make_llm_request,
    make_responses_request,
)
from llm_router.mirothinker import MiroThinkerMCPAdapter
from llm_router.openai_chat import OpenAIChatAdapter
from llm_router.responses_state import (
    ResponsesStateError,
    ResponsesStateMachine,
    iter_sse_events,
)
from llm_router.session_store import SessionStore

logger = logging.getLogger(__name__)

# ── Globals (set on app creation) ───────────────────────────────────────────

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

    model_type: "mcp_first" | "responses" | "chat"
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
            "chat",
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
    host = urlparse(base_url.rstrip("/")).netloc.lower()
    return host == "api.deepseek.com"


def _has_local_responses_session(previous_response_id: str | None) -> bool:
    if not previous_response_id or _sessions is None:
        return False
    return _sessions.get(previous_response_id) is not None


def _chat_adapter_for(upstream_name: str, base_url: str) -> DeepSeekChatAdapter:
    if _is_deepseek_upstream(upstream_name, base_url):
        return DeepSeekChatAdapter(
            forward_compat_prompt_cache=not _is_official_deepseek_base_url(
                base_url,
            ),
        )
    return OpenAIChatAdapter()


def _should_try_responses_passthrough(
    model_type: str,
    upstream_name: str,
    base_url: str,
    previous_response_id: str | None,
) -> bool:
    if model_type != "responses":
        return False
    if _has_local_responses_session(previous_response_id):
        return False
    if not _is_deepseek_upstream(upstream_name, base_url):
        return False
    return not _is_official_deepseek_base_url(base_url)


def _passthrough_response_to_client_payload(
    passthrough_response: dict[str, Any],
    requested_model: str,
) -> dict[str, Any]:
    response = dict(passthrough_response)
    response["model"] = requested_model
    return response


def _int_or_zero(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _extract_cached_input_tokens(raw_usage: dict[str, Any]) -> int:
    for details_key in ("input_tokens_details", "prompt_tokens_details"):
        details = raw_usage.get(details_key)
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
            if cached_tokens is not None:
                return _int_or_zero(cached_tokens)

    for usage_key in (
        "prompt_cache_hit_tokens",
        "cached_input_tokens",
    ):
        cached_tokens = raw_usage.get(usage_key)
        if cached_tokens is not None:
            return _int_or_zero(cached_tokens)

    return 0


def _extract_reasoning_tokens(raw_usage: dict[str, Any]) -> int:
    details = raw_usage.get("output_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            return _int_or_zero(reasoning_tokens)

    for usage_key in ("reasoning_tokens", "reasoning_output_tokens"):
        reasoning_tokens = raw_usage.get(usage_key)
        if reasoning_tokens is not None:
            return _int_or_zero(reasoning_tokens)

    return 0


def _responses_usage_from_provider(raw_usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _int_or_zero(
        raw_usage.get("input_tokens", raw_usage.get("prompt_tokens", 0)),
    )
    output_tokens = _int_or_zero(
        raw_usage.get("output_tokens", raw_usage.get("completion_tokens", 0)),
    )
    total_tokens = _int_or_zero(
        raw_usage.get("total_tokens", input_tokens + output_tokens),
    )
    cached_tokens = _extract_cached_input_tokens(raw_usage)
    reasoning_tokens = _extract_reasoning_tokens(raw_usage)

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    }


def _extract_mode_markers(content: str) -> list[str]:
    """Extract collaboration-mode names from one text blob in source order."""
    return [
        match.group("collab") or match.group("simple")
        for match in re.finditer(
            r"# Collaboration Mode:\s*(?P<collab>\w+)|# (?P<simple>Plan|Default) Mode\b",
            content,
        )
    ]


def _extract_collaboration_mode(
    messages: list[dict[str, Any]],
    instructions: str = "",
) -> str | None:
    """Find the active collaboration mode, preferring current-turn instructions."""
    if isinstance(instructions, str) and instructions:
        instruction_modes = _extract_mode_markers(instructions)
        if instruction_modes:
            return instruction_modes[-1]

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        modes = _extract_mode_markers(content)
        if modes:
            return modes[-1]
    return None


def _apply_provider_defaults(payload: dict[str, Any], model_type: str) -> dict[str, Any]:
    """Apply provider-specific defaults after adapter filtering."""
    if model_type == "mcp_first" and "repetition_penalty" not in payload:
        payload["repetition_penalty"] = _get_env_float(
            "DEFAULT_REPETITION_PENALTY",
            1.05,
        )
    return payload


def _command_looks_like_plan_execution(command: str) -> bool:
    """Detect obvious Plan-mode execution attempts without classifying all shell commands."""
    normalized = command.strip().lower()
    execution_markers = (
        "mkdir ",
        "touch ",
        "cp ",
        "mv ",
        "rm ",
        "git init",
        "uv pip install",
        "pip install",
        "python -m pip install",
        "python3 -m pip install",
        "npm install",
        "pnpm add",
        "yarn add",
        "poetry add",
        "cargo add",
        "conda install",
        "brew install",
        "apt install",
        "apt-get install",
        "uv venv",
        "python -m venv",
        "python3 -m venv",
    )
    return (
        ">" in normalized
        or ">>" in normalized
        or any(marker in normalized for marker in execution_markers)
    )


def _plan_mode_mutation_violation(
    collaboration_mode: str | None,
    output_items: list[dict[str, Any]],
) -> str | None:
    """Describe Plan-mode mutation violations, if any."""
    if collaboration_mode != "Plan":
        return None

    for item in output_items:
        item_type = item.get("type")
        if item_type == "custom_tool_call":
            return "Plan mode forbids mutating tool calls such as apply_patch."
        if item_type != "function_call":
            continue
        name = item.get("name")
        if name == "apply_patch":
            return "Plan mode forbids mutating tool calls such as apply_patch."
        if name != "exec_command":
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            continue
        command = parsed.get("cmd")
        if isinstance(command, str) and _command_looks_like_plan_execution(command):
            return "Plan mode forbids mutating exec_command calls."
    return None


def _validate_plan_mode_output_items(
    collaboration_mode: str | None,
    output_items: list[dict[str, Any]],
) -> None:
    """Reject tool calls that obviously violate Plan mode's non-mutation rule."""
    violation = _plan_mode_mutation_violation(collaboration_mode, output_items)
    if violation:
        raise ResponsesStateError(violation, "plan_mode_violation")


def _looks_like_plaintext_clarifying_question(text: str) -> bool:
    """Heuristic for Plan-mode prompts that should have used request_user_input."""
    normalized = text.strip()
    if not normalized:
        return False
    if normalized.endswith(("?", "？", ":", "：")):
        return True
    question_markers = (
        "第一个问题",
        "第二个问题",
        "第三个问题",
        "下一个问题",
        "接下来我需要",
        "我需要搞清",
    )
    return any(marker in normalized for marker in question_markers)


def _plan_mode_should_retry_with_request_user_input(
    collaboration_mode: str | None,
    output_items: list[dict[str, Any]],
    response_message: dict[str, Any],
) -> bool:
    """Detect Plan-mode plain-text questioning that should be retried via tool call."""
    if collaboration_mode != "Plan":
        return False
    if response_message.get("tool_calls"):
        return False
    if [item.get("type") for item in output_items] != ["message"]:
        return False
    content = response_message.get("content")
    if not isinstance(content, str):
        return False
    return _looks_like_plaintext_clarifying_question(content)


def _append_plan_mode_retry_feedback(
    payload: dict[str, Any],
    response_message: dict[str, Any],
) -> None:
    """Append one corrective retry turn asking the model to use request_user_input."""
    content = response_message.get("content", "")
    if content:
        payload.setdefault("messages", []).append({
            "role": "assistant",
            "content": content,
        })
    payload.setdefault("messages", []).append({
        "role": "user",
        "content": (
            "In Plan mode, clarifying questions must use the request_user_input tool "
            "instead of plain assistant text. Re-emit your latest question as exactly "
            "one request_user_input tool call."
        ),
    })


def _append_plan_mode_proposed_plan_feedback(
    payload: dict[str, Any],
    response_message: dict[str, Any],
) -> None:
    """Append one corrective retry turn asking the model to emit proposed_plan."""
    content = response_message.get("content", "")
    if content:
        payload.setdefault("messages", []).append({
            "role": "assistant",
            "content": content,
        })
    payload.setdefault("messages", []).append({
        "role": "user",
        "content": (
            "You are still in Plan mode. Do not write files, create directories, or "
            "call mutating tools. If the design is decision complete, emit the final "
            "approved plan as exactly one <proposed_plan>...</proposed_plan> block so "
            "the client can exit Plan mode and start execution. Otherwise continue "
            "planning with non-mutating actions only."
        ),
    })


def _responses_request_diagnostics(
    data: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
    model_type: str,
    upstream_name: str,
) -> dict[str, Any]:
    """Summarize request state that helps debug mode/tool-call behavior."""
    collaboration_mode = _extract_collaboration_mode(
        messages,
        data.get("instructions", ""),
    )
    request_user_input_available = None
    if collaboration_mode == "Plan":
        request_user_input_available = True
    elif collaboration_mode == "Default":
        request_user_input_available = False

    tool_names = [
        tool.get("name", tool.get("type", "unknown"))
        for tool in data.get("tools", [])
        if isinstance(tool, dict)
    ]
    return {
        "endpoint": "/v1/responses",
        "model": model,
        "model_type": model_type,
        "upstream": upstream_name,
        "collaboration_mode": collaboration_mode,
        "request_user_input_available": request_user_input_available,
        "has_previous_response_id": bool(data.get("previous_response_id")),
        "input_message_count": len(messages),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
    }


def _responses_response_diagnostics(
    llm_response: dict[str, Any],
    output_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize provider response shape for debugging router decisions."""
    choice = llm_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []
    return {
        "finish_reason": choice.get("finish_reason"),
        "has_tool_calls": bool(tool_calls),
        "tool_call_names": [
            tool_call.get("function", {}).get("name")
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ],
        "output_item_types": [item.get("type") for item in output_items],
    }


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
    usage: dict[str, Any],
) -> Response:
    """Build SSE streaming response in Responses API format."""
    return Response(
        stream_with_context(iter_sse_events(response_id, output_items, usage)),
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
        instructions = data.get("instructions", "")
        tools_raw = data.get("tools", [])
        client_requested_stream = data.get("stream", False)
        previous_response_id = data.get("previous_response_id")

        model_type, upstream_name, base_url, api_key, upstream_model = _resolve(
            model,
            data,
            None,
        )
        if _should_try_responses_passthrough(
            model_type,
            upstream_name,
            base_url,
            previous_response_id,
        ):
            passthrough_payload = dict(data)
            passthrough_payload["model"] = upstream_model
            passthrough_payload["stream"] = False
            try:
                passthrough_response = make_responses_request(
                    passthrough_payload,
                    base_url,
                    api_key,
                )
                passthrough_usage = passthrough_response.get("usage", {})
                response = _passthrough_response_to_client_payload(
                    passthrough_response,
                    model,
                )
                log_debug("CLIENT_RESPONSE /v1/responses", {
                    "status": "passthrough_success",
                    "model": model,
                    "model_type": model_type,
                    "upstream": upstream_name,
                    "usage": passthrough_usage,
                })
                if client_requested_stream:
                    return _build_sse_response(
                        response["id"],
                        response.get("output", []),
                        passthrough_usage,
                    )
                return jsonify(response), 200
            except ResponsesPassthroughUnsupportedError as exc:
                log_debug("RESPONSES_PASSTHROUGH_FALLBACK", {
                    "model": model,
                    "upstream": upstream_name,
                    "reason": str(exc),
                })

        # Convert tools to Chat format for internal use
        tools = [convert_chat_tool_to_responses(t) for t in tools_raw]

        state_machine = ResponsesStateMachine(_sessions)
        turn = state_machine.ingest_request(data, model)
        provisional_adapter = DeepSeekChatAdapter()
        messages = turn.to_chat_messages(provisional_adapter.flatten_response_items)
        model_type, upstream_name, base_url, api_key, upstream_model = _resolve(model, data, messages)
        is_deepseek = _is_deepseek_upstream(upstream_name, base_url)
        chat_adapter = _chat_adapter_for(upstream_name, base_url)
        tool_type_map = chat_adapter.tool_type_map(tools_raw)
        inject_mcp = (model_type == "mcp_first")
        if is_deepseek:
            chat_adapter.load_provider_state(
                turn.session.provider_state.get("deepseek"),
            )
        messages = turn.to_chat_messages(chat_adapter.flatten_response_items)

        # Prepend instructions as system message
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

        # Build payload
        payload = {
            k: v for k, v in data.items()
            if k not in ("messages", "tools", "input", "instructions",
                         "previous_response_id", "stream", "store", "include")
        }
        payload["model"] = upstream_model
        payload["messages"] = messages
        payload["stream"] = False  # Always non-streaming internally
        if tools_raw and not inject_mcp:
            chat_tools = (
                chat_adapter.responses_tools_to_chat(tools_raw)
                if is_deepseek
                else [convert_responses_tool_to_chat(t) for t in tools_raw]
            )
            if chat_tools:
                payload["tools"] = chat_tools
        payload = chat_adapter.filter_request_payload(payload)
        payload = _apply_provider_defaults(payload, model_type)

        # ── Run LLM ──
        llm_response, parse_result, retry_count, response_text = (
            _run_llm_with_rollback(payload, base_url, api_key, tools, inject_mcp)
        )

        # ── Build response ──
        response_id = turn.response_id
        choice = llm_response.get("choices", [{}])[0]
        response_message = choice.get("message", {})
        output_items, output_text, native_tool_calls = (
            chat_adapter.chat_response_to_output_items(
                response_message,
                tool_type_map,
            )
        )

        # Convert usage
        raw_usage = llm_response.get("usage", {})
        usage = _responses_usage_from_provider(raw_usage)

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

        plan_mode_mutation = _plan_mode_mutation_violation(
            collaboration_mode,
            output_items,
        )
        if plan_mode_mutation:
            log_debug("PLAN_MODE_PROPOSED_PLAN_RETRY", {
                "model": model,
                "upstream": upstream_name,
                "violation": plan_mode_mutation,
                "response_preview": (response_message.get("content") or "")[:240],
            })
            _append_plan_mode_proposed_plan_feedback(payload, response_message)
            llm_response = make_llm_request(payload, base_url, api_key)
            choice = llm_response.get("choices", [{}])[0]
            response_message = choice.get("message", {})
            output_items, output_text, native_tool_calls = (
                chat_adapter.chat_response_to_output_items(
                    response_message,
                    tool_type_map,
                )
            )
            raw_usage = llm_response.get("usage", {})
            usage = _responses_usage_from_provider(raw_usage)
            tool_calls_list = native_tool_calls or []

        _validate_plan_mode_output_items(collaboration_mode, output_items)

        if _plan_mode_should_retry_with_request_user_input(
            collaboration_mode,
            output_items,
            response_message,
        ):
            log_debug("PLAN_MODE_REQUEST_USER_INPUT_RETRY", {
                "model": model,
                "upstream": upstream_name,
                "response_preview": response_message.get("content", "")[:240],
            })
            _append_plan_mode_retry_feedback(payload, response_message)
            llm_response = make_llm_request(payload, base_url, api_key)
            choice = llm_response.get("choices", [{}])[0]
            response_message = choice.get("message", {})
            output_items, output_text, native_tool_calls = (
                chat_adapter.chat_response_to_output_items(
                    response_message,
                    tool_type_map,
                )
            )
            raw_usage = llm_response.get("usage", {})
            usage = _responses_usage_from_provider(raw_usage)
            tool_calls_list = native_tool_calls or []
            _validate_plan_mode_output_items(collaboration_mode, output_items)

        has_tool_output = bool(tool_calls_list)
        output_text = None if has_tool_output else output_text
        provider_state_updates = (
            {"deepseek": chat_adapter.dump_provider_state()}
            if is_deepseek
            else None
        )
        log_debug(
            "RESPONSES_RESPONSE_DIAGNOSTICS",
            _responses_response_diagnostics(llm_response, output_items),
        )
        state_machine.commit_response(
            turn,
            output_items,
            provider_state_updates,
        )

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
            "session_items": len(turn.session.items),
            "usage": usage,
        })

        if client_requested_stream:
            return _build_sse_response(response_id, output_items, usage)

        return jsonify(response), 200

    except ResponsesStateError as e:
        log_debug("CLIENT_RESPONSE /v1/responses", {
            "status": "state_error",
            "error_code": e.code,
            "error": e.message,
        })
        return jsonify(e.to_error_dict()), e.status_code

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
