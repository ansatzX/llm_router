"""Responses endpoint regression tests."""

from unittest.mock import Mock

import llm_router.server as server_mod
from llm_router.config import UpstreamConfig
from llm_router.llm_client import LLMRequestError
from llm_router.session_store import SessionStore
from tests.responses._helpers import _configure_test_app


def test_responses_chat_route_resumes_pending_tool_call_state(
    tmp_path,
    monkeypatch,
):
    """Chat routes should replay pending tool state without text transcripts."""
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
    server_mod._config.default_model_type = "responses_chat"
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
            "role": "assistant",
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
        {
            "role": "tool",
            "tool_call_id": "call_ls",
            "content": "README.md\nsrc",
        },
    ]
    assert "historical tool call omitted" not in str(second_payload["messages"])
    assert "Tool output:" not in str(second_payload["messages"])


def test_responses_deepseek_web_search_failure_does_not_commit_session(
    tmp_path,
    monkeypatch,
):
    """Failed DeepSeek hosted search requests must not advance local state."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    def failing_bridge_run(
        self,
        *,
        messages,
        tools_raw,
        max_tokens=None,
        temperature=None,
        top_p=None,
        request_options=None,
    ):
        raise LLMRequestError("bridge failed", status_code=503)

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.DeepSeekAnthropicWebSearchBridge.run",
        failing_bridge_run,
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
            "input": "Find current docs",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 502
    assert response.get_json()["error"]["code"] == "provider_error"
    assert len(server_mod._sessions) == 0
    assert mock_make_request.call_count == 0


def test_responses_deepseek_malformed_web_search_result_does_not_commit_session(
    tmp_path,
    monkeypatch,
):
    """Malformed hosted search payloads must fail before local session commit."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    def malformed_request(payload, llm_base_url, api_key):
        return {
            "stop_reason": "end_turn",
            "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        malformed_request,
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
            "input": "Find current docs",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 502
    assert response.get_json()["error"]["code"] == "provider_error"
    assert len(server_mod._sessions) == 0
    assert mock_make_request.call_count == 0


def test_responses_deepseek_anthropic_bridge_replays_thinking_blocks(
    tmp_path,
    monkeypatch,
):
    """DeepSeek Anthropic thinking blocks must survive previous_response_id replay."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        if len(requests) == 1:
            return {
                "stop_reason": "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "identify the right Cunxi Gong"},
                    {"type": "text", "text": "First answer."},
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Second answer."}],
            "usage": {"input_tokens": 5, "output_tokens": 6},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    first = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "搜索一下，谁是Cunxi Gong",
            "tools": [{"type": "web_search", "external_web_access": True}],
            "reasoning": {"effort": "high"},
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first.get_json()["id"],
            "input": "继续",
            "tools": [{"type": "web_search", "external_web_access": True}],
            "reasoning": {"effort": "high"},
        },
    )

    assert second.status_code == 200
    assert mock_make_request.call_count == 0
    assert requests[1]["messages"] == [
        {"role": "user", "content": "搜索一下，谁是Cunxi Gong"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "identify the right Cunxi Gong"},
                {"type": "text", "text": "First answer."},
            ],
        },
        {"role": "user", "content": "继续"},
    ]


def test_responses_deepseek_anthropic_bridge_recovers_thinking_for_stateless_tool_replay(
    tmp_path,
    monkeypatch,
):
    """Hosted web_search bridge thinking sidecars must survive full-history replay."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        if len(requests) == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "thinking", "thinking": "inspect files before answer"},
                    {
                        "type": "tool_use",
                        "id": "call_ls",
                        "name": "exec_command",
                        "input": {"cmd": "ls"},
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 5, "output_tokens": 6},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    first = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "look around",
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "function", "name": "exec_command"},
            ],
            "reasoning": {"effort": "high"},
        },
    )
    assert first.status_code == 200

    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "look around"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_ls",
                    "name": "exec_command",
                    "arguments": '{"cmd": "ls"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_ls",
                    "output": "README.md",
                },
            ],
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "function", "name": "exec_command"},
            ],
            "reasoning": {"effort": "high"},
        },
    )

    assert second.status_code == 200
    assert mock_make_request.call_count == 0
    assistant_tool_messages = [
        message for message in requests[1]["messages"]
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(
            block.get("type") == "tool_use"
            for block in message.get("content", [])
            if isinstance(block, dict)
        )
    ]
    assert len(assistant_tool_messages) == 1
    assert assistant_tool_messages[0]["content"][0] == {
        "type": "thinking",
        "thinking": "inspect files before answer",
    }


def test_responses_deepseek_anthropic_bridge_recovers_thinking_after_store_reload(
    tmp_path,
    monkeypatch,
):
    """Hosted web_search bridge thinking sidecars must be durable on disk."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        if len(requests) == 1:
            return {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "thinking", "thinking": "durable bridge thinking"},
                    {
                        "type": "tool_use",
                        "id": "call_read",
                        "name": "exec_command",
                        "input": {"cmd": "cat README.md"},
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        return {
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 5, "output_tokens": 6},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    first = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "read docs",
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "function", "name": "exec_command"},
            ],
            "reasoning": {"effort": "high"},
        },
    )
    assert first.status_code == 200

    store_path = server_mod._sessions.store_path
    server_mod._sessions = SessionStore(store_path=store_path, ttl_seconds=3600)
    reloaded_session = server_mod._sessions.get(first.get_json()["id"])
    assert reloaded_session.provider_state["deepseek"]["reasoning_by_call_id"] == {
        "call_read": "durable bridge thinking",
    }

    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "read docs"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_read",
                    "name": "exec_command",
                    "arguments": '{"cmd": "cat README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_read",
                    "output": "# llm_router",
                },
            ],
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {"type": "function", "name": "exec_command"},
            ],
            "reasoning": {"effort": "high"},
        },
    )

    assert second.status_code == 200
    assert mock_make_request.call_count == 0
    assistant_tool_messages = [
        message for message in requests[1]["messages"]
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(
            block.get("type") == "tool_use"
            for block in message.get("content", [])
            if isinstance(block, dict)
        )
    ]
    assert len(assistant_tool_messages) == 1
    assert assistant_tool_messages[0]["content"][0] == {
        "type": "thinking",
        "thinking": "durable bridge thinking",
    }


def test_responses_deepseek_anthropic_missing_thinking_error_is_client_visible(
    tmp_path,
    monkeypatch,
):
    """DeepSeek Anthropic thinking replay 400 should not surface as generic provider_error."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    def failing_bridge_run(
        self,
        *,
        messages,
        tools_raw,
        max_tokens=None,
        temperature=None,
        top_p=None,
        request_options=None,
    ):
        raise LLMRequestError(
            "The `content[].thinking` in the thinking mode must be passed back to the API.",
            status_code=400,
        )

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.DeepSeekAnthropicWebSearchBridge.run",
        failing_bridge_run,
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
            "input": "搜索一下，谁是Cunxi Gong",
            "tools": [{"type": "web_search", "external_web_access": True}],
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 409
    assert mock_make_request.call_count == 0
    body = response.get_json()
    assert body["error"]["code"] == "deepseek_missing_reasoning_content"
    assert "content[].thinking" in body["error"]["message"]


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
    server_mod._config.default_model_type = "responses_chat"
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

def test_responses_deepseek_legacy_parallel_tool_calls_keep_interleaved_side_channel(
    tmp_path,
    monkeypatch,
):
    """DeepSeek replay should keep structured tool messages around side channels."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 124,
            "choices": [{"message": {"content": "opened"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
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
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "open site"}],
                },
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"first"}',
                    "call_id": "call_first",
                },
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": '{"cmd":"second"}',
                    "call_id": "call_second",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_first",
                    "output": "first output",
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
                    "call_id": "call_second",
                    "output": "second output",
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
        "tool",
        "system",
    ]
    assert payload["messages"][1]["tool_calls"][0]["id"] == "call_first"
    assert payload["messages"][1]["tool_calls"][1]["id"] == "call_second"
    assert payload["messages"][2]["tool_call_id"] == "call_first"
    assert payload["messages"][3]["tool_call_id"] == "call_second"
    assert "historical tool call omitted" not in str(payload["messages"])
    assert "Tool output:" not in str(payload["messages"])

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
    server_mod._config.default_model_type = "responses_chat"
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
        }
    }
    store.save()
    server_mod._deepseek_adapter.reset()
    server_mod._config.default_model_type = "responses_chat"
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

def test_responses_deepseek_missing_reasoning_provider_error_is_client_visible(
    tmp_path,
    monkeypatch,
):
    """DeepSeek's thinking replay 400 should not be hidden as a 500."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
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
    store.save()
    original_items = list(session.items)
    mock_make_request.side_effect = LLMRequestError(
        "The `reasoning_content` in the thinking mode must be passed back to the API.",
        status_code=400,
        body={
            "error": {
                "message": (
                    "The `reasoning_content` in the thinking mode must be "
                    "passed back to the API."
                ),
                "type": "invalid_request_error",
                "code": "invalid_request_error",
            }
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
            "previous_response_id": session.response_id,
            "input": "continue",
        },
    )

    assert response.status_code == 409
    body = response.get_json()
    assert body["error"]["code"] == "deepseek_missing_reasoning_content"
    assert "call_ls" in body["error"]["message"]
    assert server_mod._sessions.get(session.response_id).items == original_items

def test_responses_deepseek_recovers_reasoning_for_stateless_codex_replay(
    tmp_path,
    monkeypatch,
):
    """Codex can resend full history without previous_response_id."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "checking files",
                        "reasoning_content": "inspect repo before editing",
                        "tool_calls": [
                            {
                                "id": "call_ls",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"ls"}',
                                },
                            },
                            {
                                "id": "call_pyproject",
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": '{"cmd":"cat pyproject.toml"}',
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
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    first = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "look around"},
    )

    assert first.status_code == 200
    mock_make_request.return_value = {
        "created": 124,
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "look around"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_ls",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_pyproject",
                    "name": "exec_command",
                    "arguments": '{"cmd":"cat pyproject.toml"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_ls",
                    "output": "README.md",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_pyproject",
                    "output": "[project]",
                },
            ],
        },
    )

    assert second.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assistant_tool_messages = [
        message for message in payload["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(assistant_tool_messages) == 1
    assert (
        assistant_tool_messages[0]["reasoning_content"]
        == "inspect repo before editing"
    )

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
    server_mod._config.default_model_type = "responses_chat"
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

def test_responses_deepseek_plain_message_reasoning_is_not_content_keyed_sidecar(
    tmp_path,
    monkeypatch,
):
    """Plain assistant reasoning should stay on the item, not in a text-keyed cache."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "same visible text",
                        "reasoning_content": "first private reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
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
        json={"model": "test-model", "input": "reply with repeated text"},
    )

    assert response.status_code == 200
    session = server_mod._sessions.get(response.get_json()["id"])
    assert session.items[-1]["reasoning_content"] == "first private reasoning"
    assert session.provider_state["deepseek"] == {"reasoning_by_call_id": {}}

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
    server_mod._config.default_model_type = "responses_chat"
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
    server_mod._config.default_model_type = "responses_chat"
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


def test_responses_xiaomi_persists_provider_reasoning_state(tmp_path, monkeypatch):
    """Xiaomi thinking tool calls should persist provider-private reasoning."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "need tool",
                        "tool_calls": [
                            {
                                "id": "call_xiaomi",
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
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://api.xiaomimimo.com",
    )
    server_mod._config.default_upstream = "xiaomi"

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "input": "list files",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "parameters": {"type": "object"},
                }
            ],
        },
    )

    assert response.status_code == 200
    session = next(iter(server_mod._sessions._store.values()))
    assert session.provider_state["xiaomi"]["reasoning_by_call_id"] == {
        "call_xiaomi": "need tool",
    }


def test_responses_xiaomi_replays_provider_reasoning_state(tmp_path, monkeypatch):
    """Xiaomi tool replay should use Xiaomi provider_state, not DeepSeek state."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
    )
    server_mod._config.default_upstream = "xiaomi"

    session = server_mod._sessions.create("mimo-v2.5-pro")
    session.items = [
        {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd":"ls"}',
            "call_id": "call_xiaomi",
        },
        {
            "type": "function_call_output",
            "call_id": "call_xiaomi",
            "output": "README.md",
        },
    ]
    session.provider_state = {
        "xiaomi": {
            "reasoning_by_call_id": {"call_xiaomi": "persisted xiaomi reasoning"},
        },
        "deepseek": {
            "reasoning_by_call_id": {"call_xiaomi": "wrong provider reasoning"},
        },
    }
    server_mod._sessions.save()

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "previous_response_id": session.response_id,
            "input": "continue",
        },
    )

    assert response.status_code == 200
    messages = mock_make_request.call_args.args[0]["messages"]
    assert messages[0]["reasoning_content"] == "persisted xiaomi reasoning"
