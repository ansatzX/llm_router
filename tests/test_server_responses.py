"""Endpoint tests for Responses API routing and session behavior."""

from unittest.mock import Mock

import llm_router.server as server_mod
from llm_router.config import RouterConfig, UpstreamConfig
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
    monkeypatch.setattr(server_mod, "make_llm_request", mock_make_request)
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
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }
    ]


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
