"""Responses endpoint regression tests."""

import llm_router.server as server_mod
from llm_router.config import RouteConfig, UpstreamConfig
from tests.responses._helpers import _configure_test_app


def test_responses_chat_route_forwards_codex_tools_as_chat_tools(tmp_path, monkeypatch):
    """Plain chat routes must preserve Codex function tools for upstream models."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "list files",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command",
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            ],
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "Runs a command",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        },
    ]
    assert payload["tool_choice"] == "auto"

def test_responses_deepseek_route_filters_responses_metadata(
    tmp_path,
    monkeypatch,
):
    """DeepSeek adapter decides which Responses fields can reach Chat API."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "client_metadata": {"x-codex-installation-id": "install-id"},
            "prompt_cache_key": "cache-key",
            "text": {"format": {"type": "json_schema"}},
            "reasoning": None,
            "temperature": 0.7,
            "repetition_penalty": 1.05,
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert "client_metadata" not in payload
    assert "prompt_cache_key" not in payload
    assert "text" not in payload
    assert "reasoning" not in payload
    assert "repetition_penalty" not in payload
    assert payload["temperature"] == 0.7

def test_responses_default_chat_provider_filters_metadata_and_maps_text_format(
    tmp_path,
    monkeypatch,
):
    """OpenAI-compatible chat providers should not receive Responses-only fields."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["default"] = UpstreamConfig(
        base_url="https://aihubmix.com/v1",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4-mini",
            "input": "hi",
            "client_metadata": {"x-codex-installation-id": "install-id"},
            "prompt_cache_key": "cache-key",
            "text": {"format": {"type": "json_schema", "name": "out"}},
            "reasoning": None,
            "temperature": 0.7,
            "repetition_penalty": 1.05,
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert "client_metadata" not in payload
    assert "prompt_cache_key" not in payload
    assert "text" not in payload
    assert "reasoning" not in payload
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"type": "json_schema", "name": "out"},
    }
    assert "repetition_penalty" not in payload
    assert payload["temperature"] == 0.7

def test_responses_default_chat_provider_supports_fast_and_plan_fields(
    tmp_path,
    monkeypatch,
):
    """OpenAI-compatible chat providers should receive mapped fast/plan controls."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["default"] = UpstreamConfig(
        base_url="https://aihubmix.com/v1",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4-mini",
            "input": "hi",
            "service_tier": "priority",
            "reasoning": {"effort": "medium", "summary": "auto"},
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["service_tier"] == "priority"
    assert payload["reasoning_effort"] == "medium"
    assert "reasoning" not in payload

def test_responses_can_rewrite_requested_model_to_provider_model(
    tmp_path,
    monkeypatch,
):
    """Router should preserve client-facing model slug while rewriting upstream slug."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.routes = [
        RouteConfig(
            pattern="gpt-*",
            model_type="responses_chat",
            upstream="deepseek",
            upstream_model="deepseek-v4-flash",
        )
    ]

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.4-mini", "input": "hi"},
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["model"] == "deepseek-v4-flash"
    assert response.get_json()["model"] == "gpt-5.4-mini"

def test_responses_passthrough_route_uses_provider_responses_api(
    tmp_path,
    monkeypatch,
):
    """responses_passthrough routes are explicit provider-owned Responses calls."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_responses_request = server_mod.make_responses_request
    mock_make_responses_request.return_value = {
        "id": "resp_gateway",
        "object": "response",
        "created": 123,
        "model": "provider-deployment",
        "output": [
            {
                "id": "msg_gateway",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
                "status": "completed",
            }
        ],
        "output_text": "ok",
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 9},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 15,
        },
        "status": "completed",
    }
    server_mod._config.upstreams["aicc0"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com/v1",
    )
    server_mod._config.routes = [
        RouteConfig(
            pattern="deepseek-gateway-*",
            model_type="responses_passthrough",
            upstream="aicc0",
            upstream_model="deepseek-v4-pro",
        )
    ]

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-gateway-pro",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "Bash",
                    "description": "Run a shell command.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                }
            ],
            "prompt_cache_key": "cache-key",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    passthrough_payload, base_url, _api_key = mock_make_responses_request.call_args.args
    assert base_url == "https://zapi.aicc0.com/v1"
    assert passthrough_payload["model"] == "deepseek-v4-pro"
    assert passthrough_payload["stream"] is False
    assert passthrough_payload["prompt_cache_key"] == "cache-key"
    assert passthrough_payload["tools"] == [
        {
            "type": "function",
            "name": "Bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ]
    assert b"response.output_item.added" in response.data
    assert b"response.output_text.delta" in response.data
    assert b"response.completed" in response.data

def test_responses_route_does_not_auto_passthrough_non_official_deepseek(
    tmp_path,
    monkeypatch,
):
    """Non-official DeepSeek-compatible URLs need an explicit passthrough route."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["deepseek_gateway"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com/v1",
    )
    server_mod._config.routes = [
        RouteConfig(
            pattern="deepseek-gateway-*",
            model_type="responses",
            upstream="deepseek_gateway",
        )
    ]

    response = client.post(
        "/v1/responses",
        json={"model": "deepseek-gateway-pro", "input": "hi"},
    )

    assert response.status_code == 200
    assert server_mod.make_responses_request.call_count == 0
    assert mock_make_request.call_count == 1

def test_responses_passthrough_route_never_calls_official_deepseek_responses(
    tmp_path,
    monkeypatch,
):
    """Official DeepSeek must stay on the Chat adapter path."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.routes = [
        RouteConfig(
            pattern="deepseek-*",
            model_type="responses_passthrough",
            upstream="deepseek",
            upstream_model="deepseek-chat",
        )
    ]

    response = client.post(
        "/v1/responses",
        json={"model": "deepseek-chat", "input": "hi"},
    )

    assert response.status_code == 200
    assert server_mod.make_responses_request.call_count == 0
    assert mock_make_request.call_count == 1
    assert mock_make_request.call_args.args[0]["model"] == "deepseek-chat"

def test_responses_passthrough_continuation_failure_does_not_fallback_locally(
    tmp_path,
    monkeypatch,
):
    """Provider-owned previous_response_id cannot be recovered by local state."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_responses_request = server_mod.make_responses_request
    mock_make_responses_request.side_effect = RuntimeError("gateway unavailable")
    server_mod._config.upstreams["aicc0"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com/v1",
    )
    server_mod._config.routes = [
        RouteConfig(
            pattern="deepseek-gateway-*",
            model_type="responses_passthrough",
            upstream="aicc0",
            upstream_model="deepseek-v4-pro",
        )
    ]

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-gateway-pro",
            "previous_response_id": "resp_provider_owned",
            "input": "continue",
        },
    )

    assert response.status_code == 502
    assert response.get_json()["error"]["type"] == "provider_error"
    assert mock_make_responses_request.call_count == 1
    assert mock_make_request.call_count == 0

def test_responses_memory_phase_one_request_rewrites_only_fixed_small_model(
    tmp_path,
    monkeypatch,
):
    """Memory phase-1 prompt should override gpt-5.4-mini to DeepSeek flash."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["default"] = UpstreamConfig(
        base_url="https://aihubmix.com/v1",
    )
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.4-mini",
            "instructions": "## Memory Writing Agent: Phase 1 (Single Rollout)",
            "input": (
                "Analyze this rollout and produce JSON with `raw_memory`, "
                "`rollout_summary`, and `rollout_slug` (use empty string when unknown)."
            ),
        },
    )

    assert response.status_code == 200
    payload, base_url = mock_make_request.call_args.args[:2]
    assert payload["model"] == "deepseek-v4-flash"
    assert base_url == "https://api.deepseek.com"
    assert response.get_json()["model"] == "gpt-5.4-mini"

def test_responses_memory_phase_two_headers_rewrite_gpt_5_4_to_deepseek_pro(
    tmp_path,
    monkeypatch,
):
    """Memory consolidation headers should reroute gpt-5.4 without affecting user turns."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["default"] = UpstreamConfig(
        base_url="https://aihubmix.com/v1",
    )
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        headers={
            "x-openai-memgen-request": "true",
            "x-openai-subagent": "memory_consolidation",
        },
        json={"model": "gpt-5.4", "input": "consolidate memories"},
    )

    assert response.status_code == 200
    payload, base_url = mock_make_request.call_args.args[:2]
    assert payload["model"] == "deepseek-v4-pro"
    assert base_url == "https://api.deepseek.com"
    assert response.get_json()["model"] == "gpt-5.4"

def test_responses_regular_gpt_5_4_request_is_not_misclassified_as_memory(
    tmp_path,
    monkeypatch,
):
    """Interactive gpt-5.4 turns should keep their normal upstream routing."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.upstreams["default"] = UpstreamConfig(
        base_url="https://aihubmix.com/v1",
    )
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.4", "input": "review this code"},
    )

    assert response.status_code == 200
    payload, base_url = mock_make_request.call_args.args[:2]
    assert payload["model"] == "gpt-5.4"
    assert base_url == "https://aihubmix.com/v1"
