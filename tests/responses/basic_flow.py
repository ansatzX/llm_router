"""Responses endpoint regression tests."""

import llm_router.server as server_mod
from tests.responses._helpers import _configure_test_app


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

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "hi", "stream": True},
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
