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
