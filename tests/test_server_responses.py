"""Endpoint tests for Responses API routing and session behavior."""

from unittest.mock import Mock

import llm_router.server as server_mod
from llm_router.config import RouteConfig, RouterConfig, UpstreamConfig
from llm_router.session_store import SessionStore


def _configure_test_app(tmp_path, monkeypatch, llm_response):
    cfg = RouterConfig(
        upstreams={"default": UpstreamConfig(base_url="http://backend.test/v1")},
        routes=[],
        default_model_type="responses",
        default_upstream="default",
    )
    server_mod._config = cfg
    server_mod._sessions = SessionStore(
        store_path=tmp_path / "sessions.json",
        ttl_seconds=3600,
    )
    mock_make_request = Mock(return_value=llm_response)
    mock_make_responses_request = Mock()
    monkeypatch.setattr(server_mod, "make_llm_request", mock_make_request)
    monkeypatch.setattr(
        server_mod,
        "make_responses_request",
        mock_make_responses_request,
    )
    server_mod.app.config.update(TESTING=True)
    return server_mod.app.test_client(), mock_make_request


def test_responses_previous_response_id_continues_same_session(tmp_path, monkeypatch):
    """The response ID returned by one turn links to the same session next turn."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    first = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first"}],
                }
            ],
        },
    ).get_json()

    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first["id"],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "second"}],
                }
            ],
        },
    ).get_json()

    assert second["id"] != first["id"]
    second_messages = mock_make_request.call_args_list[1].args[0]["messages"]
    assert [m["content"] for m in second_messages] == ["first", "ok", "second"]


def test_responses_returns_standard_output_items(tmp_path, monkeypatch):
    """Responses JSON includes output array items in addition to output_text."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "Hi"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output_text"] == "hello"
    assert body["output"] == [
        {
            "id": body["output"][0]["id"],
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ]
    assert body["output"][0]["id"].startswith("msg_")


def test_responses_returns_reasoning_content_on_message_items(tmp_path, monkeypatch):
    """DeepSeek non-tool replies can also carry reasoning_content."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "hello",
                        "reasoning_content": "plain response reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"][0]["reasoning_content"] == "plain response reasoning"


def test_responses_developer_role_is_forwarded_as_system(tmp_path, monkeypatch):
    """Developer messages are normalized for Chat Completions backends."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "follow policy"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
        },
    )

    assert response.status_code == 200
    messages = mock_make_request.call_args.args[0]["messages"]
    assert messages[0] == {"role": "system", "content": "follow policy"}
    assert messages[1] == {"role": "user", "content": "hello"}


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
    server_mod._config.default_model_type = "chat"
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
    server_mod._config.default_model_type = "chat"
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


def test_responses_deepseek_gateway_can_forward_prompt_cache_key(
    tmp_path,
    monkeypatch,
):
    """DeepSeek-compatible gateways may rely on OpenAI prompt-cache fields."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "prompt_cache_key": "cache-key",
            "prompt_cache_retention": {"type": "persistent"},
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["prompt_cache_key"] == "cache-key"
    assert payload["prompt_cache_retention"] == {"type": "persistent"}


def test_responses_non_official_deepseek_uses_native_responses_passthrough(
    tmp_path,
    monkeypatch,
):
    """Compatible DeepSeek gateways should preserve native Responses behavior."""
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
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "deepseek"
    mock_make_responses_request.return_value = {
        "id": "resp_passthrough",
        "object": "response",
        "created": 123,
        "model": "accounts/demo/deployments/abc",
        "output": [
            {
                "id": "msg_ok",
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

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "input": "hi",
            "prompt_cache_key": "cache-key",
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    passthrough_payload = mock_make_responses_request.call_args.args[0]
    assert passthrough_payload["model"] == "deepseek-v4-pro"
    assert passthrough_payload["stream"] is False
    assert passthrough_payload["prompt_cache_key"] == "cache-key"
    assert b"response.completed" in response.data
    assert b"cached_tokens" in response.data


def test_responses_passthrough_forwards_previous_response_id(
    tmp_path,
    monkeypatch,
):
    """Native passthrough must preserve provider-managed response threading."""
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
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "deepseek"
    mock_make_responses_request.side_effect = [
        {
            "id": "resp_first",
            "object": "response",
            "created": 123,
            "model": "accounts/demo/deployments/abc",
            "output": [
                {
                    "id": "msg_first",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "first"}],
                    "status": "completed",
                }
            ],
            "output_text": "first",
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
            "status": "completed",
        },
        {
            "id": "resp_second",
            "object": "response",
            "created": 124,
            "model": "accounts/demo/deployments/abc",
            "output": [
                {
                    "id": "msg_second",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "second"}],
                    "status": "completed",
                }
            ],
            "output_text": "second",
            "usage": {
                "input_tokens": 14,
                "input_tokens_details": {"cached_tokens": 9},
                "output_tokens": 4,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 18,
            },
            "status": "completed",
        },
    ]

    first = client.post(
        "/v1/responses",
        json={"model": "deepseek-v4-pro", "input": "first"},
    )
    second = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "previous_response_id": "resp_first",
            "input": "second",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert mock_make_request.call_count == 0
    assert mock_make_responses_request.call_count == 2
    second_payload = mock_make_responses_request.call_args_list[1].args[0]
    assert second_payload["previous_response_id"] == "resp_first"
    assert second.get_json()["usage"]["input_tokens_details"]["cached_tokens"] == 9


def test_responses_passthrough_falls_back_to_chat_emulation_when_unsupported(
    tmp_path,
    monkeypatch,
):
    """Non-compatible gateways should transparently fall back to local emulation."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_responses_request = server_mod.make_responses_request
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "deepseek"
    mock_make_responses_request.side_effect = (
        server_mod.ResponsesPassthroughUnsupportedError("unsupported")
    )

    response = client.post(
        "/v1/responses",
        json={"model": "deepseek-v4-pro", "input": "hi"},
    )

    assert response.status_code == 200
    assert mock_make_responses_request.call_count == 1
    assert mock_make_request.call_count == 1


def test_responses_passthrough_falls_back_to_chat_emulation_when_transport_fails(
    tmp_path,
    monkeypatch,
):
    """Transient passthrough transport failures should not leak 500s to Codex."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_responses_request = server_mod.make_responses_request
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://zapi.aicc0.com",
    )
    server_mod._config.default_model_type = "responses"
    server_mod._config.default_upstream = "deepseek"
    mock_make_responses_request.side_effect = (
        server_mod.ResponsesPassthroughTransientError("proxy reset")
    )

    response = client.post(
        "/v1/responses",
        json={"model": "deepseek-v4-pro", "input": "hi"},
    )

    assert response.status_code == 200
    assert mock_make_responses_request.call_count == 1
    assert mock_make_request.call_count == 1


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
            model_type="chat",
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


def test_responses_request_diagnostics_prefers_latest_plan_mode_block(
    tmp_path,
    monkeypatch,
):
    """Diagnostics must detect current Plan mode even if older Default text remains."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    logged = []

    def _capture_log(event, data=None, truncate=True):  # noqa: ARG001
        logged.append((event, data))

    monkeypatch.setattr(server_mod, "log_debug", _capture_log)
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "instructions": (
                "<collaboration_mode># Plan Mode (Conversational)\n\n"
                "You are in **Plan Mode** until a developer message explicitly ends it.\n"
                "</collaboration_mode>"
            ),
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "<collaboration_mode># Collaboration Mode: Default\n\n"
                                "`request_user_input` tool is unavailable in Default mode\n"
                                "</collaboration_mode>"
                            ),
                        }
                    ],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "plan this"}],
                },
            ],
        },
    )

    assert response.status_code == 200
    request_diag = next(
        data for event, data in logged
        if event == "RESPONSES_REQUEST_DIAGNOSTICS"
    )
    assert request_diag["collaboration_mode"] == "Plan"
    assert request_diag["request_user_input_available"] is True


def test_responses_rejects_mutating_exec_command_tool_call_in_plan_mode(
    tmp_path,
    monkeypatch,
):
    """Plan mode may explore, but should not execute install/write commands."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_install",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": (
                                        '{"cmd":"uv pip install openai --quiet"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "instructions": (
                "<collaboration_mode># Plan Mode (Conversational)\n\n"
                "You are in **Plan Mode** until a developer message explicitly ends it.\n"
                "</collaboration_mode>"
            ),
            "input": "plan this",
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "plan_mode_violation"


def test_responses_plan_mode_retries_plain_question_as_request_user_input(
    tmp_path,
    monkeypatch,
):
    """Plan mode should retry once when the model asks a plain-text question instead of the tool."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "明白了，做任务编排与自动化。接下来我需要搞清几个关键维度。\n\n第二个问题：",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_request.side_effect = [
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "明白了，做任务编排与自动化。接下来我需要搞清几个关键维度。\n\n第二个问题：",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {
            "created": 124,
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_question_2",
                                "type": "function",
                                "function": {
                                    "name": "request_user_input",
                                    "arguments": (
                                        '{"questions":[{"header":"Task kind","id":"task_kind",'
                                        '"question":"这个 multi-agent system 主要处理哪类任务？",'
                                        '"options":[{"label":"任务编排","description":"聚焦调度与自动化。"}]}]}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    ]
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "instructions": (
                "<collaboration_mode># Plan Mode (Conversational)\n\n"
                "You are in **Plan Mode** until a developer message explicitly ends it.\n"
                "</collaboration_mode>"
            ),
            "input": "想一想怎么做一个mutli agent system",
            "tools": [
                {
                    "type": "function",
                    "name": "request_user_input",
                    "description": "Ask one question",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {"type": "array"},
                        },
                        "required": ["questions"],
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"][0]["type"] == "function_call"
    assert body["output"][0]["name"] == "request_user_input"

    assert mock_make_request.call_count == 2
    retry_payload = mock_make_request.call_args_list[1].args[0]
    assert retry_payload["messages"][-2]["role"] == "assistant"
    assert "第二个问题" in retry_payload["messages"][-2]["content"]
    assert retry_payload["messages"][-1]["role"] == "user"
    assert "request_user_input" in retry_payload["messages"][-1]["content"]


def test_responses_plan_mode_retries_mutating_exec_command_as_proposed_plan(
    tmp_path,
    monkeypatch,
):
    """Plan mode should steer mutating exec_command responses back to proposed_plan output."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "好，进入写 spec 文件阶段。先创建 docs 目录。",
                        "tool_calls": [
                            {
                                "id": "call_mkdir",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": (
                                        '{"cmd":"mkdir -p /Users/ansatz/data/code/search_test/docs/superpowers/specs"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_request.side_effect = [
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "好，进入写 spec 文件阶段。先创建 docs 目录。",
                        "tool_calls": [
                            {
                                "id": "call_mkdir",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": (
                                        '{"cmd":"mkdir -p /Users/ansatz/data/code/search_test/docs/superpowers/specs"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {
            "created": 124,
            "choices": [
                {
                    "message": {
                        "content": (
                            "<proposed_plan>\n"
                            "## Multi-agent system spec\n\n"
                            "- Write the approved spec after leaving Plan mode.\n"
                            "</proposed_plan>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    ]
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "instructions": (
                "<collaboration_mode># Plan Mode (Conversational)\n\n"
                "You are in **Plan Mode** until a developer message explicitly ends it.\n"
                "</collaboration_mode>"
            ),
            "input": "继续完成最后的 spec 输出",
            "tools": [
                {"type": "function", "name": "exec_command"},
                {"type": "function", "name": "apply_patch"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"][0]["type"] == "message"
    assert "<proposed_plan>" in body["output"][0]["content"][0]["text"]

    assert mock_make_request.call_count == 2
    retry_payload = mock_make_request.call_args_list[1].args[0]
    assert retry_payload["messages"][-2]["role"] == "assistant"
    assert "写 spec 文件" in retry_payload["messages"][-2]["content"]
    assert retry_payload["messages"][-1]["role"] == "user"
    assert "<proposed_plan>" in retry_payload["messages"][-1]["content"]


def test_responses_plan_mode_retries_apply_patch_as_proposed_plan(
    tmp_path,
    monkeypatch,
):
    """Plan mode should steer apply_patch responses back to proposed_plan output."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "现在写 spec 文件。",
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": (
                                        '{"input":"*** Begin Patch\\n*** Add File: docs/superpowers/specs/spec.md\\n+draft\\n*** End Patch\\n"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_request.side_effect = [
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "现在写 spec 文件。",
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": (
                                        '{"input":"*** Begin Patch\\n*** Add File: docs/superpowers/specs/spec.md\\n+draft\\n*** End Patch\\n"}'
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {
            "created": 124,
            "choices": [
                {
                    "message": {
                        "content": (
                            "<proposed_plan>\n"
                            "## Spec handoff\n\n"
                            "- Emit the approved plan and wait for execute mode.\n"
                            "</proposed_plan>"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    ]
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "deepseek-v4-pro",
            "instructions": (
                "<collaboration_mode># Plan Mode (Conversational)\n\n"
                "You are in **Plan Mode** until a developer message explicitly ends it.\n"
                "</collaboration_mode>"
            ),
            "input": "把最终 plan 收束一下",
            "tools": [
                {"type": "function", "name": "apply_patch"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"][0]["type"] == "message"
    assert "<proposed_plan>" in body["output"][0]["content"][0]["text"]

    assert mock_make_request.call_count == 2
    retry_payload = mock_make_request.call_args_list[1].args[0]
    assert retry_payload["messages"][-2]["role"] == "assistant"
    assert "写 spec 文件" in retry_payload["messages"][-2]["content"]
    assert retry_payload["messages"][-1]["role"] == "user"
    assert "<proposed_plan>" in retry_payload["messages"][-1]["content"]


def test_responses_logs_collaboration_mode_and_completion_diagnostics(
    tmp_path,
    monkeypatch,
):
    """Responses logs should expose mode and completion details for debugging."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    logged = []

    def _capture_log(event, data=None, truncate=True):  # noqa: ARG001
        logged.append((event, data))

    monkeypatch.setattr(server_mod, "log_debug", _capture_log)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "instructions": (
                "<collaboration_mode># Collaboration Mode: Default\n\n"
                "`request_user_input` tool is unavailable in Default mode\n"
                "</collaboration_mode>"
            ),
            "input": "think about multi agents",
        },
    )

    assert response.status_code == 200

    request_diag = next(
        data for event, data in logged
        if event == "RESPONSES_REQUEST_DIAGNOSTICS"
    )
    assert request_diag["collaboration_mode"] == "Default"
    assert request_diag["request_user_input_available"] is False
    assert request_diag["input_message_count"] == 2

    response_diag = next(
        data for event, data in logged
        if event == "RESPONSES_RESPONSE_DIAGNOSTICS"
    )
    assert response_diag["finish_reason"] == "stop"
    assert response_diag["has_tool_calls"] is False
    assert response_diag["output_item_types"] == ["message"]


def test_responses_chat_route_resumes_pending_tool_call_state(
    tmp_path,
    monkeypatch,
):
    """Chat routes should use router state when previous_response_id is provided."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_ls",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    first = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "list files",
            "tools": [{"type": "function", "name": "exec_command"}],
        },
    )

    assert first.status_code == 200
    first_body = first.get_json()
    assert first_body["output"][0]["call_id"] == "call_ls"

    mock_make_request.return_value = {
        "created": 124,
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first_body["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_ls",
                    "output": "README.md\nsrc",
                }
            ],
            "tools": [{"type": "function", "name": "exec_command"}],
        },
    )

    assert second.status_code == 200
    second_payload = mock_make_request.call_args.args[0]
    assert second_payload["messages"] == [
        {"role": "user", "content": "list files"},
        {
            "role": "user",
            "content": "[historical tool call omitted: exec_command call_id=call_ls]\n"
            '{"cmd":"ls"}\nTool output:\nREADME.md\nsrc',
        },
    ]


def test_responses_accepts_codex_side_channel_between_tool_call_and_output(
    tmp_path,
    monkeypatch,
):
    """Codex may include approval messages before the matching tool output."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 124,
            "choices": [{"message": {"content": "opened"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "open site"}],
                },
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"open \\"https://chat.deepseek.com/\\""}',
                    "call_id": "call_open",
                    "reasoning_content": "open the web app",
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": 'Approved command prefix saved:\n- ["open"]',
                        }
                    ],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_open",
                    "output": "Process exited with code 0",
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert [message["role"] for message in payload["messages"]] == [
        "user",
        "assistant",
        "tool",
        "system",
    ]


def test_responses_deepseek_persists_provider_reasoning_state(
    tmp_path,
    monkeypatch,
):
    """DeepSeek reasoning cache should survive in session provider_state."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "run ls first",
                        "tool_calls": [
                            {
                                "id": "call_ls",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "list files"},
    )

    assert response.status_code == 200
    session = server_mod._sessions.get(response.get_json()["id"])
    assert session.provider_state["deepseek"]["reasoning_by_call_id"] == {
        "call_ls": "run ls first",
    }


def test_responses_deepseek_restores_reasoning_from_provider_state_after_restart(
    tmp_path,
    monkeypatch,
):
    """Persisted provider_state should rebuild DeepSeek thinking history."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    store = server_mod._sessions
    session = store.create("test-model")
    session.items = [
        {
            "type": "function_call",
            "id": "call_ls",
            "call_id": "call_ls",
            "name": "exec_command",
            "arguments": '{"cmd":"ls"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_ls",
            "output": "README.md",
        },
    ]
    session.pending_tool_calls = {
        "call_ls": {
            "call_id": "call_ls",
            "name": "exec_command",
            "type": "function_call",
            "arguments": '{"cmd":"ls"}',
            "created_response_id": session.response_id,
            "status": "satisfied",
        }
    }
    session.provider_state = {
        "deepseek": {
            "reasoning_by_call_id": {"call_ls": "persisted reasoning"},
            "reasoning_by_message_content": {},
        }
    }
    store.save()
    server_mod._deepseek_adapter.reset()
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": session.response_id,
            "input": "continue",
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["messages"][0]["role"] == "assistant"
    assert payload["messages"][0]["reasoning_content"] == "persisted reasoning"


def test_responses_deepseek_provider_state_survives_session_store_reload(
    tmp_path,
    monkeypatch,
):
    """Provider sidecar must be durable, not only present on live session objects."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "persisted after reload",
                        "tool_calls": [
                            {
                                "id": "call_reload",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    store_path = server_mod._sessions.store_path
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "list files"},
    )

    reloaded_store = SessionStore(store_path=store_path, ttl_seconds=3600)
    reloaded_session = reloaded_store.get(response.get_json()["id"])
    assert reloaded_session.provider_state["deepseek"]["reasoning_by_call_id"] == {
        "call_reload": "persisted after reload",
    }


def test_responses_does_not_mutate_session_when_upstream_fails(
    tmp_path,
    monkeypatch,
):
    """Input/tool-output state is committed only after an upstream response."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning_content": "run ls",
                        "tool_calls": [
                            {
                                "id": "call_ls",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    first = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "list files"},
    ).get_json()
    session = server_mod._sessions.get(first["id"])
    original_items = list(session.items)

    mock_make_request.side_effect = RuntimeError("upstream down")
    failed = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_ls",
                    "output": "README.md",
                }
            ],
        },
    )

    assert failed.status_code == 500
    session_after_failure = server_mod._sessions.get(first["id"])
    assert session_after_failure.items == original_items
    assert (
        session_after_failure.pending_tool_calls["call_ls"]["status"]
        == "pending"
    )

    mock_make_request.side_effect = None
    mock_make_request.return_value = {
        "created": 124,
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    retry = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_ls",
                    "output": "README.md",
                }
            ],
        },
    )

    assert retry.status_code == 200


def test_responses_does_not_create_empty_session_when_new_turn_fails(
    tmp_path,
    monkeypatch,
):
    """A failed first turn should not leave an empty persisted session."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    monkeypatch.setattr(
        server_mod,
        "make_llm_request",
        Mock(side_effect=RuntimeError("upstream down")),
    )

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hello"},
    )

    assert response.status_code == 500
    assert len(server_mod._sessions) == 0


def test_responses_non_deepseek_route_does_not_receive_deepseek_reasoning_cache(
    tmp_path,
    monkeypatch,
):
    """DeepSeek provider-private state must not leak into other providers."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    session = server_mod._sessions.create("test-model")
    session.items = [
        {
            "type": "function_call",
            "id": "call_cached",
            "call_id": "call_cached",
            "name": "exec_command",
            "arguments": '{"cmd":"ls"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_cached",
            "output": "README.md",
        },
    ]
    session.pending_tool_calls = {
        "call_cached": {
            "call_id": "call_cached",
            "name": "exec_command",
            "type": "function_call",
            "arguments": '{"cmd":"ls"}',
            "created_response_id": session.response_id,
            "status": "satisfied",
        }
    }
    server_mod._sessions.save()
    server_mod._deepseek_adapter.record_tool_reasoning(
        "call_cached",
        "leaked reasoning",
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": session.response_id,
            "input": "continue",
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert all("reasoning_content" not in message for message in payload["messages"])


def test_responses_rejects_duplicate_tool_call_ids_without_persisting(
    tmp_path,
    monkeypatch,
):
    """Duplicate provider call IDs should not corrupt pending state."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_dup",
                                "type": "function",
                                "function": {"name": "first", "arguments": "{}"},
                            },
                            {
                                "id": "call_dup",
                                "type": "function",
                                "function": {"name": "second", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "call tools"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "duplicate_tool_call"
    assert len(server_mod._sessions) == 0


def test_responses_rejects_unknown_tool_output_before_upstream(
    tmp_path,
    monkeypatch,
):
    """Unknown tool outputs are state errors and must not reach a provider."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "missing_call",
                    "output": "orphaned output",
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_tool_output"
    mock_make_request.assert_not_called()


def test_responses_rejects_partial_parallel_tool_outputs_before_upstream(
    tmp_path,
    monkeypatch,
):
    """Parallel tool calls must all be satisfied before the next model call."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_one",
                                "type": "function",
                                "function": {"name": "first", "arguments": "{}"},
                            },
                            {
                                "id": "call_two",
                                "type": "function",
                                "function": {"name": "second", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    first = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "call both tools"},
    )
    assert first.status_code == 200
    first_body = first.get_json()

    mock_make_request.reset_mock()
    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first_body["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_one",
                    "output": "one done",
                }
            ],
        },
    )

    assert second.status_code == 400
    assert second.get_json()["error"]["code"] == "pending_tool_calls"
    mock_make_request.assert_not_called()


def test_responses_chat_route_wraps_non_function_tools_for_chat_backend(
    tmp_path,
    monkeypatch,
):
    """Chat backends must still see Codex custom tools as callable functions."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "edit file",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "freeform patch",
                    "format": {"type": "grammar"},
                },
                {"type": "web_search", "external_web_access": False},
            ],
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["tools"][0] == (
        {
            "type": "function",
            "function": {
                "name": "exec_command",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    )
    assert payload["tools"][1]["type"] == "function"
    assert payload["tools"][1]["function"]["name"] == "apply_patch"
    assert "freeform patch" in payload["tools"][1]["function"]["description"]
    assert payload["tools"][1]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Freeform input for the custom tool.",
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }
    assert payload["tools"][2]["type"] == "function"
    assert payload["tools"][2]["function"]["name"] == "web_search"


def test_responses_chat_route_returns_chat_tool_calls_as_response_items(
    tmp_path,
    monkeypatch,
):
    """Native Chat tool calls from DeepSeek should be returned as Responses items."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls -la"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "list files"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output_text"] is None
    assert body["output"] == [
        {
            "type": "function_call",
            "id": "call_abc",
            "call_id": "call_abc",
            "name": "exec_command",
            "arguments": '{"cmd":"ls -la"}',
        }
    ]


def test_responses_chat_route_restores_custom_tool_calls_as_response_items(
    tmp_path,
    monkeypatch,
):
    """Wrapped custom tool calls should be restored for Codex execution."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"input":"*** Begin Patch\\n*** End Patch\\n"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "edit",
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "freeform patch",
                    "format": {"type": "grammar"},
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"] == [
        {
            "type": "custom_tool_call",
            "id": "call_patch",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch\n",
        }
    ]


def test_responses_stream_restores_custom_tool_calls_as_response_items(
    tmp_path,
    monkeypatch,
):
    """Streaming Responses events must preserve Codex custom tool item types."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"input":"*** Begin Patch\\n*** End Patch\\n"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "edit",
            "stream": True,
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "freeform patch",
                    "format": {"type": "grammar"},
                }
            ],
        },
    )

    assert response.status_code == 200
    event_stream = response.get_data(as_text=True)
    assert '"type": "custom_tool_call"' in event_stream
    assert '"input": "*** Begin Patch\\n*** End Patch\\n"' in event_stream


def test_responses_chat_route_returns_reasoning_content_on_tool_call_items(
    tmp_path,
    monkeypatch,
):
    """DeepSeek thinking mode requires reasoning_content to round-trip."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "I should list files first.",
                        "tool_calls": [
                            {
                                "id": "call_reasoning",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls -la"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "chat"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "list files"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"][0]["reasoning_content"] == "I should list files first."


def test_responses_mirothinker_route_uses_mcp_prompt_not_native_tools(
    tmp_path,
    monkeypatch,
):
    """MiroThinker is MCP-first: tools are prompt-injected, not forwarded."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": """
<use_mcp_tool>
  <server_name>tools</server_name>
  <tool_name>exec_command</tool_name>
  <arguments>{"cmd":"ls"}</arguments>
</use_mcp_tool>
""",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "mcp_first"
    server_mod._config.default_upstream = "mirothinker"
    server_mod._config.upstreams["mirothinker"] = UpstreamConfig(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
    )

    response = client.post(
        "/v1/responses",
        json={
            "model": "mirothinker-test",
            "input": "list files",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Runs a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
            "stream": True,
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert "tools" not in payload
    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "<use_mcp_tool>" in payload["messages"][0]["content"]
    event_stream = response.get_data(as_text=True)
    assert '"type": "function_call"' in event_stream
    assert '"name": "exec_command"' in event_stream
