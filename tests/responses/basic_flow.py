"""Responses endpoint regression tests."""

import json

import llm_router.server as server_mod
from llm_router.config import UpstreamConfig
from llm_router.llm_client import LLMRequestError
from tests.responses._helpers import _configure_test_app


def _sse_payloads(body: str, event_type: str) -> list[dict]:
    payloads = []
    current_event = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
            continue
        if line.startswith("data: ") and current_event == event_type:
            payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


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

def test_responses_returns_reasoning_output_item(tmp_path, monkeypatch):
    """DeepSeek reasoning_content should become a Responses reasoning item."""
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
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"][0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "plain response reasoning"}],
        "content": [{"type": "reasoning_text", "text": "plain response reasoning"}],
    }
    assert body["output"][1]["type"] == "message"


def test_responses_stream_emits_reasoning_deltas(tmp_path, monkeypatch):
    """Simulated SSE should expose reasoning deltas for Codex clients."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "done",
                        "reasoning_content": "plain response reasoning",
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
        json={
            "model": "test-model",
            "input": "hi",
            "stream": True,
            "instructions": "# Plan Mode",
        },
    )

    assert response.status_code == 200
    assert b"response.reasoning_summary_part.added" in response.data
    assert b"response.reasoning_summary_text.delta" in response.data
    assert b"response.reasoning_text.delta" in response.data

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


def test_responses_stream_uses_real_upstream_streaming_for_text_and_reasoning(
    tmp_path,
    monkeypatch,
):
    """responses_chat text turns should proxy upstream stream=True deltas."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    seen_stream_flag = {"called": False}

    def fake_stream(payload, _base_url, _api_key):
        seen_stream_flag["called"] = True
        yield {
            "choices": [
                {"delta": {"reasoning_content": "plan "}, "finish_reason": None},
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {"delta": {"reasoning_content": "step"}, "finish_reason": None},
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {"delta": {"content": "Hel"}, "finish_reason": None},
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {"delta": {"content": "lo"}, "finish_reason": "stop"},
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "stream": True},
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert seen_stream_flag["called"] is True
    assert mock_make_request.call_count == 0

    assert '"type": "response.reasoning_summary_text.delta"' in body
    assert '"delta": "plan "' in body
    assert '"delta": "step"' in body
    assert '"type": "response.output_text.delta"' in body
    assert '"delta": "Hel"' in body
    assert '"delta": "lo"' in body
    assert '"type": "response.completed"' in body
    assert server_mod._sessions.stats()["session_count"] == 1


def test_responses_stream_failure_emits_failed_and_does_not_commit(
    tmp_path,
    monkeypatch,
):
    """Mid-stream upstream failures must not advance local Responses state."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"

    def failing_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [{"delta": {"content": "partial"}, "finish_reason": None}],
            "usage": None,
        }
        raise LLMRequestError("upstream stream aborted", status_code=502)

    monkeypatch.setattr(server_mod, "make_llm_stream_request", failing_stream)

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "stream": True},
    )

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert '"type": "response.failed"' in body
    assert "upstream stream aborted" in body
    assert server_mod._sessions.stats()["session_count"] == 0


def test_responses_stream_deepseek_tools_emit_custom_tool_deltas(
    tmp_path,
    monkeypatch,
):
    """DeepSeek tool-call chunks should stream as Responses tool events."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(payload, _base_url, _api_key):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"input":"*** Begin Patch\\n',
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "*** End Patch\\n\"}",
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "patch it",
            "stream": True,
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "freeform patch",
                }
            ],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    added = _sse_payloads(body, "response.output_item.added")
    assert added[0]["item"]["type"] == "custom_tool_call"
    assert added[0]["item"]["input"] == ""
    assert '"type": "response.custom_tool_call_input.delta"' in body
    assert "*** Begin Patch\\n*** End Patch\\n" in body
    assert '"type": "response.completed"' in body

    # commit-after-success: successful stream creates one persisted session
    assert server_mod._sessions.stats()["session_count"] == 1


def test_responses_stream_custom_tool_extracts_input_before_full_json(
    tmp_path,
    monkeypatch,
):
    """Custom tool input deltas should not wait for the whole wrapped JSON."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_patch",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"input":"*** Begin',
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": ' Patch\\nline'},
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "patch it",
            "stream": True,
            "tools": [{"type": "custom", "name": "apply_patch"}],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    deltas = _sse_payloads(body, "response.custom_tool_call_input.delta")
    assert [event["delta"] for event in deltas[:2]] == [
        "*** Begin",
        " Patch\nline",
    ]
    assert '"type": "response.completed"' in body


def test_responses_stream_deepseek_tools_delayed_id_and_name_replay_buffered_args(
    tmp_path,
    monkeypatch,
):
    """When DeepSeek sends arguments before id/name, stream should keep consistency."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"location":"Sh'},
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'anghai","unit":"c"}'},
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "function": {"name": "get_weather"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather?",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    added = _sse_payloads(body, "response.output_item.added")
    assert added[0]["item"]["id"] == "call_weather"
    assert added[0]["item"]["arguments"] == ""
    assert '"type": "response.function_call_arguments.delta"' in body
    assert '\\"location\\":\\"Sh' in body
    assert 'anghai\\",\\"unit\\":\\"c\\"}' in body
    assert '"type": "response.completed"' in body


def test_responses_stream_deepseek_tools_support_parallel_calls_without_index(
    tmp_path,
    monkeypatch,
):
    """Tool stream assembly should tolerate mixed indexed/non-indexed parallel deltas."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '{"input":"*** Begin Patch\\n',
                                },
                            },
                            {
                                "index": 0,
                                "id": "call_weather",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"Sha',
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_patch",
                                "function": {"arguments": '*** End Patch\\n"}'},
                            },
                            {
                                "index": 0,
                                "id": "call_weather",
                                "function": {"arguments": 'nghai"}'},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "run tools",
            "stream": True,
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "freeform patch",
                },
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            ],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    added = _sse_payloads(body, "response.output_item.added")
    assert [
        (event["item"]["call_id"], event["item"].get("input"), event["item"].get("arguments"))
        for event in added
    ] == [
        ("call_patch", "", None),
        ("call_weather", None, ""),
    ]
    assert '"call_id": "call_patch"' in body
    assert '"call_id": "call_weather"' in body
    assert '"type": "response.custom_tool_call_input.delta"' in body
    assert "*** Begin Patch\\n*** End Patch\\n" in body
    assert '"type": "response.function_call_arguments.delta"' in body
    assert '\\"location\\":\\"Sha' in body
    assert 'nghai\\"}' in body
    assert '"type": "response.completed"' in body


def test_responses_stream_tool_added_and_done_order_stay_consistent(
    tmp_path,
    monkeypatch,
):
    """Tool item lifecycle should keep one stable order even with delayed metadata."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"a":1'},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {
                                    "name": "tool_b",
                                    "arguments": '{"b":2}',
                                },
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "tool_a", "arguments": "}"},
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "run tools",
            "stream": True,
            "tools": [
                {"type": "function", "name": "tool_a", "parameters": {"type": "object"}},
                {"type": "function", "name": "tool_b", "parameters": {"type": "object"}},
            ],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    added = _sse_payloads(body, "response.output_item.added")
    done = _sse_payloads(body, "response.output_item.done")
    assert [
        (event["output_index"], event["item"]["call_id"])
        for event in added
    ] == [(0, "call_b"), (1, "call_a")]
    assert [
        (event["output_index"], event["item"]["call_id"])
        for event in done
    ] == [(0, "call_b"), (1, "call_a")]


def test_responses_stream_late_reasoning_after_tool_fails_without_commit(
    tmp_path,
    monkeypatch,
):
    """Reasoning that arrives after a visible output item would corrupt indexes."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_tool",
                                "function": {"name": "tool_a", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": None,
                },
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {"reasoning_content": "late reasoning"},
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "run tool",
            "stream": True,
            "tools": [
                {"type": "function", "name": "tool_a", "parameters": {"type": "object"}},
            ],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '"type": "response.failed"' in body
    assert "Late streamed reasoning after output items is unsupported" in body
    assert server_mod._sessions.stats()["session_count"] == 0


def test_responses_stream_mixed_text_and_tool_can_reorder_when_enabled(
    tmp_path,
    monkeypatch,
):
    """Experimental mixed-stream mode should buffer later tools instead of failing."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"
    monkeypatch.setenv("LLM_ROUTER_EXPERIMENTAL_MIXED_STREAM", "1")

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {"delta": {"content": "I will call a tool."}, "finish_reason": None},
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"Shanghai"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather?",
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }
            ],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '"type": "response.failed"' not in body
    assert '"type": "response.output_text.delta"' in body
    assert '"type": "response.function_call_arguments.delta"' in body
    assert '"type": "response.completed"' in body


def test_responses_stream_mixed_text_and_tool_fails_by_default(
    tmp_path,
    monkeypatch,
):
    """Mixed-stream support remains opt-in until Codex e2e coverage exists."""
    client, _ = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    def fake_stream(_payload, _base_url, _api_key):
        yield {
            "choices": [
                {"delta": {"content": "I will call a tool."}, "finish_reason": None},
            ],
            "usage": None,
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"Shanghai"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

    monkeypatch.setattr(server_mod, "make_llm_stream_request", fake_stream)

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "weather?",
            "stream": True,
            "tools": [{"type": "function", "name": "get_weather"}],
        },
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '"type": "response.failed"' in body
    assert "Mixed streamed tool_calls with streamed text is unsupported" in body
    assert server_mod._sessions.stats()["session_count"] == 0
