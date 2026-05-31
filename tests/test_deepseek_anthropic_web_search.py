"""Tests for DeepSeek Anthropic hosted web-search bridge."""

from __future__ import annotations

import pytest

from llm_router.deepseek.anthropic_web_search import (
    _MAX_PAUSE_TURN_RETRIES,
    DeepSeekAnthropicSearchExecution,
    DeepSeekAnthropicWebSearchBridge,
    DeepSeekAnthropicWebSearchExecutor,
    _anthropic_messages_url,
    _anthropic_tool_from_codex_web_search,
    _messages_from_chat_messages,
    _queries_from_dsml_text,
    _queries_from_internal_tool_arguments,
    _usage_from_anthropic_response,
    make_deepseek_anthropic_messages_request,
)
from llm_router.llm_client import LLMRequestError


def test_anthropic_messages_url_targets_messages_endpoint():
    assert (
        _anthropic_messages_url("https://api.deepseek.com")
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        _anthropic_messages_url("https://api.deepseek.com/v1")
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        _anthropic_messages_url("https://api.deepseek.com/anthropic")
        == "https://api.deepseek.com/anthropic/v1/messages"
    )


def test_codex_web_search_maps_to_supported_anthropic_tool_fields():
    tool = _anthropic_tool_from_codex_web_search(
        {
            "type": "web_search",
            "search_context_size": "high",
            "filters": {"allowed_domains": ["example.com"]},
            "user_location": {
                "type": "approximate",
                "country": "US",
                "region": "CA",
                "city": "San Francisco",
                "timezone": "America/Los_Angeles",
            },
            "max_uses": 5,
            "unknown": "ignored",
        }
    )

    assert tool == {
        "type": "web_search_20250305",
        "name": "web_search",
        "search_context_size": "high",
        "allowed_domains": ["example.com"],
        "user_location": {
            "type": "approximate",
            "country": "US",
            "region": "CA",
            "city": "San Francisco",
            "timezone": "America/Los_Angeles",
        },
    }


def test_codex_web_search_drops_unverified_options():
    tool = _anthropic_tool_from_codex_web_search(
        {
            "type": "web_search",
            "search_context_size": "massive",
            "allowed_domains": ["unverified.example"],
            "blocked_domains": ["blocked.example"],
            "max_uses": 5,
        }
    )

    assert tool == {"type": "web_search_20250305", "name": "web_search"}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            '{"query":" first ","queries":["second","first","third","second"]}',
            ["first", "second", "third"],
        ),
        (
            {
                "query": ["first", "second", "first"],
                "queries": "third",
            },
            ["first", "second", "third"],
        ),
    ],
)
def test_queries_from_internal_tool_arguments_deduplicate_in_order(arguments, expected):
    assert _queries_from_internal_tool_arguments(arguments) == expected


def test_queries_from_dsml_text_extracts_web_search_queries_in_order():
    text = (
        '<｜｜DSML｜｜tool_calls>'
        '<｜｜DSML｜｜invoke name="web_search">{"query":"first"}'
        '<｜｜DSML｜｜invoke name="web_search">{"queries":["second","first","third"]}'
        '<｜｜DSML｜｜invoke name="other_tool">{"query":"ignored"}'
    )

    assert _queries_from_dsml_text(text) == ["first", "second", "third"]


def test_queries_from_dsml_text_extracts_raw_parameter_blocks():
    text = (
        '<｜｜DSML｜｜tool_calls>'
        '<｜｜DSML｜｜invoke name="web_search">'
        '<｜｜DSML｜｜parameter name="query" string="true">Zhigang Shuai latest paper 2026</｜｜DSML｜｜parameter>'
        '</｜｜DSML｜｜invoke>'
        '</｜｜DSML｜｜tool_calls>'
    )

    assert _queries_from_dsml_text(text) == ["Zhigang Shuai latest paper 2026"]


def test_executor_runs_minimal_web_search_request_and_parses_search_results(monkeypatch):
    executor = DeepSeekAnthropicWebSearchExecutor(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srv_1",
                    "name": "web_search",
                    "input": {"query": "Rust tutorials"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv_1",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://www.rust-lang.org/learn",
                            "title": "Rust Learn",
                            "encrypted_content": "enc",
                            "page_age": "April 30, 2025",
                        }
                    ],
                },
                {"type": "text", "text": "Learn Rust at rust-lang.org."},
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    result = executor.execute(["Rust tutorials"])

    assert len(requests) == 1
    assert requests[0] == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "Rust tutorials"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "max_tokens": 256,
    }
    assert isinstance(result, DeepSeekAnthropicSearchExecution)
    assert result.searches[0]["id"] == "srv_1"
    assert result.searches[0]["query"] == "Rust tutorials"
    assert result.searches[0]["results"] == [
        {
            "type": "web_search_result",
            "url": "https://www.rust-lang.org/learn",
            "title": "Rust Learn",
            "encrypted_content": "enc",
            "page_age": "April 30, 2025",
        }
    ]
    assert result.searches[0]["text"] == "Learn Rust at rust-lang.org."
    assert result.text == "Learn Rust at rust-lang.org."
    assert result.usage == {
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "server_tool_use": {"web_search_requests": 1},
        "total_tokens": 7,
    }
    assert result.raw_response["stop_reason"] == "end_turn"


def test_executor_combines_multiple_queries_and_usage(monkeypatch):
    executor = DeepSeekAnthropicWebSearchExecutor(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )
    queries_seen: list[str] = []

    def fake_request(payload, llm_base_url, api_key):
        query = payload["messages"][0]["content"]
        queries_seen.append(query)
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": f"srv_{len(queries_seen)}",
                    "name": "web_search",
                    "input": {"query": query},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": f"srv_{len(queries_seen)}",
                    "content": [],
                },
                {"type": "text", "text": query.upper()},
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    result = executor.execute(["first", "second"])

    assert queries_seen == ["first", "second"]
    assert [search["query"] for search in result.searches] == ["first", "second"]
    assert [search["text"] for search in result.searches] == ["FIRST", "SECOND"]
    assert result.text == "FIRST\nSECOND"
    assert result.usage == {
        "input_tokens": 2,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0},
        "server_tool_use": {"web_search_requests": 2},
        "total_tokens": 6,
    }


def test_executor_retries_dsml_follow_up_queries_and_preserves_results(monkeypatch):
    executor = DeepSeekAnthropicWebSearchExecutor(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        query = payload["messages"][0]["content"]
        if len(requests) == 1:
            return {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srv_1",
                        "name": "web_search",
                        "input": {"query": query},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv_1",
                        "content": [],
                    },
                    {
                        "type": "text",
                        "text": (
                            '<｜｜DSML｜｜tool_calls>'
                            '<｜｜DSML｜｜invoke name="web_search">'
                            '{"query":"Rust macros"}'
                        ),
                    },
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 6,
                    "server_tool_use": {"web_search_requests": 1},
                },
            }
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srv_2",
                    "name": "web_search",
                    "input": {"query": query},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv_2",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/rust-macros",
                            "title": "Rust macros",
                            "encrypted_content": "enc",
                        }
                    ],
                },
                {"type": "text", "text": "Learn Rust macros."},
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 8,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    result = executor.execute(["Rust tutorials"])

    assert len(requests) == 2
    assert requests[0]["messages"] == [{"role": "user", "content": "Rust tutorials"}]
    assert requests[1]["messages"] == [{"role": "user", "content": "Rust macros"}]
    assert [search["query"] for search in result.searches] == ["Rust tutorials", "Rust macros"]
    assert result.searches[0]["text"] is None
    assert result.searches[1]["text"] == "Learn Rust macros."
    assert result.text == "Learn Rust macros."


def test_executor_returns_partial_results_when_dsml_follow_ups_exhaust_budget(monkeypatch):
    executor = DeepSeekAnthropicWebSearchExecutor(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        query = payload["messages"][0]["content"]
        index = len(requests)
        if index == 1:
            return {
                "stop_reason": "end_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srv_1",
                        "name": "web_search",
                        "input": {"query": query},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv_1",
                        "content": [],
                    },
                    {
                        "type": "text",
                        "text": (
                            '<｜｜DSML｜｜tool_calls>'
                            '<｜｜DSML｜｜invoke name="web_search">'
                            '{"query":"Rust macros"}'
                        ),
                    },
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 6,
                    "server_tool_use": {"web_search_requests": 1},
                },
            }
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": f"srv_{index}",
                    "name": "web_search",
                    "input": {"query": query},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": f"srv_{index}",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": f"https://example.com/{index}",
                            "title": f"Result {index}",
                            "encrypted_content": "enc",
                        }
                    ],
                },
                {"type": "text", "text": f"Result {index}"},
                {
                    "type": "text",
                    "text": (
                        '<｜｜DSML｜｜tool_calls>'
                        '<｜｜DSML｜｜invoke name="web_search">'
                        f'{{"query":"{query} follow-up"}}'
                    ),
                },
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 8,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    result = executor.execute(["Rust tutorials"])

    assert len(requests) == _MAX_PAUSE_TURN_RETRIES
    assert [search["query"] for search in result.searches] == [
        "Rust tutorials",
        "Rust macros",
        "Rust macros follow-up",
        "Rust macros follow-up follow-up",
    ]
    assert result.searches[0]["results"] == []
    assert result.searches[1]["results"]
    assert result.searches[-1]["text"] == f"Result {_MAX_PAUSE_TURN_RETRIES}"
    assert result.text == "Result 2\nResult 3\nResult 4"


def test_executor_raises_when_search_budget_is_exhausted_by_empty_results(monkeypatch):
    executor = DeepSeekAnthropicWebSearchExecutor(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        query = payload["messages"][0]["content"]
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": f"srv_{len(requests)}",
                    "name": "web_search",
                    "input": {"query": query},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": f"srv_{len(requests)}",
                    "content": [],
                },
                {
                    "type": "text",
                    "text": (
                        '<｜｜DSML｜｜tool_calls>'
                        '<｜｜DSML｜｜invoke name="web_search">'
                        f'{{"query":"{query} follow-up"}}'
                    ),
                },
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    with pytest.raises(LLMRequestError, match="empty search results") as excinfo:
        executor.execute(["Rust tutorials"])

    assert excinfo.value.status_code == 502
    assert len(requests) == _MAX_PAUSE_TURN_RETRIES


def test_messages_from_chat_messages_preserves_user_assistant_and_tool_results():
    messages = _messages_from_chat_messages(
        [
            {"role": "system", "content": "system text"},
            {"role": "developer", "content": "developer note"},
            {"role": "user", "content": "find current docs"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key":"value"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "tool output",
            },
        ]
    )

    assert messages == [
        {"role": "user", "content": "find current docs"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup",
                    "input": {"key": "value"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "tool output",
                }
            ],
        },
    ]


def test_messages_from_chat_messages_groups_parallel_tool_results():
    messages = _messages_from_chat_messages(
        [
            {"role": "user", "content": "inspect repo"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "read_b", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            {"role": "user", "content": "now search"},
        ]
    )

    assert messages == [
        {"role": "user", "content": "inspect repo"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_1", "name": "read_a", "input": {}},
                {"type": "tool_use", "id": "call_2", "name": "read_b", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "a"},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "b"},
            ],
        },
        {"role": "user", "content": "now search"},
    ]


def test_bridge_runs_pause_turn_continuation_and_parses_search_blocks(monkeypatch):
    bridge = DeepSeekAnthropicWebSearchBridge(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )
    requests: list[dict[str, object]] = []

    def fake_request(payload, llm_base_url, api_key):
        requests.append(payload)
        if len(requests) == 1:
            return {
                "stop_reason": "pause_turn",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srv_1",
                        "name": "web_search",
                        "input": {"query": "Rust tutorials"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://www.rust-lang.org/learn",
                                "title": "Rust Learn",
                                "encrypted_content": "enc",
                                "page_age": "April 30, 2025",
                            }
                        ],
                    },
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 12,
                    "server_tool_use": {"web_search_requests": 1},
                },
            }
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "Learn Rust at rust-lang.org.",
                    "citations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://www.rust-lang.org/learn",
                            "title": "Rust Learn",
                            "encrypted_index": "abc",
                            "cited_text": "Rust is great.",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 13,
                "output_tokens": 14,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    result = bridge.run(
        messages=[{"role": "user", "content": "Search Rust tutorials"}],
        tools_raw=[
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["rust-lang.org"]},
                "location": {
                    "type": "approximate",
                    "country": "US",
                    "region": "CA",
                    "city": "San Francisco",
                    "timezone": "America/Los_Angeles",
                },
            }
        ],
        max_tokens=128,
        temperature=0,
    )

    assert len(requests) == 2
    assert "system" not in requests[0]
    assert requests[0]["messages"] == [{"role": "user", "content": "Search Rust tutorials"}]
    assert requests[0]["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["rust-lang.org"],
            "user_location": {
                "type": "approximate",
                "country": "US",
                "region": "CA",
                "city": "San Francisco",
                "timezone": "America/Los_Angeles",
            },
        }
    ]
    assert requests[1]["messages"][-1]["role"] == "assistant"
    assert result.output_items == [
        {
            "type": "web_search_call",
            "id": "srv_1",
            "status": "completed",
            "action": {"type": "search", "query": "Rust tutorials"},
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Learn Rust at rust-lang.org.",
                    "annotations": [
                        {
                            "type": "web_search_result_location",
                            "url": "https://www.rust-lang.org/learn",
                            "title": "Rust Learn",
                            "encrypted_index": "abc",
                            "cited_text": "Rust is great.",
                        }
                    ],
                }
            ],
        },
    ]
    assert result.output_text == "Learn Rust at rust-lang.org."
    assert result.usage == {
        "input_tokens": 24,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 26,
        "output_tokens_details": {"reasoning_tokens": 0},
        "server_tool_use": {"web_search_requests": 2},
        "total_tokens": 50,
    }


def test_bridge_rejects_dsml_residual_text(monkeypatch):
    bridge = DeepSeekAnthropicWebSearchBridge(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )

    def fake_request(payload, llm_base_url, api_key):
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="web_search">',
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    with pytest.raises(LLMRequestError, match="DSML"):
        bridge.run(
            messages=[{"role": "user", "content": "Search docs"}],
            tools_raw=[{"type": "web_search"}],
        )


def test_bridge_rejects_search_without_final_text(monkeypatch):
    bridge = DeepSeekAnthropicWebSearchBridge(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )

    def fake_request(payload, llm_base_url, api_key):
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srv_1",
                    "name": "web_search",
                    "input": {"query": "Rust tutorials"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv_1",
                    "content": [],
                },
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    with pytest.raises(LLMRequestError, match="final text"):
        bridge.run(
            messages=[{"role": "user", "content": "Search Rust tutorials"}],
            tools_raw=[{"type": "web_search"}],
        )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ([], "usable output"),
        ([{"type": "thinking", "thinking": "searching"}], "usable output"),
        (
            [
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv_1",
                    "content": [],
                }
            ],
            "usable output",
        ),
    ],
)
def test_bridge_rejects_malformed_content_without_usable_output(
    monkeypatch,
    content,
    match,
):
    bridge = DeepSeekAnthropicWebSearchBridge(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )

    def fake_request(payload, llm_base_url, api_key):
        return {
            "stop_reason": "end_turn",
            "content": content,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    with pytest.raises(LLMRequestError, match=match):
        bridge.run(
            messages=[{"role": "user", "content": "Search Rust tutorials"}],
            tools_raw=[{"type": "web_search"}],
        )


def test_bridge_rejects_web_search_tool_use_block(monkeypatch):
    bridge = DeepSeekAnthropicWebSearchBridge(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )

    def fake_request(payload, llm_base_url, api_key):
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "web_search",
                    "input": {"query": "Rust tutorials"},
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    with pytest.raises(LLMRequestError, match="server_tool_use"):
        bridge.run(
            messages=[{"role": "user", "content": "Search Rust tutorials"}],
            tools_raw=[{"type": "web_search"}],
        )


def test_bridge_rejects_web_search_call_without_id(monkeypatch):
    bridge = DeepSeekAnthropicWebSearchBridge(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-pro",
    )

    def fake_request(payload, llm_base_url, api_key):
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "server_tool_use",
                    "name": "web_search",
                    "input": {"query": "Rust tutorials"},
                },
                {"type": "text", "text": "Found results."},
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "server_tool_use": {"web_search_requests": 1},
            },
        }

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.make_deepseek_anthropic_messages_request",
        fake_request,
    )

    with pytest.raises(LLMRequestError, match="id"):
        bridge.run(
            messages=[{"role": "user", "content": "Search Rust tutorials"}],
            tools_raw=[{"type": "web_search"}],
        )


def test_usage_from_anthropic_response_preserves_server_tool_usage():
    usage = _usage_from_anthropic_response(
        {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "server_tool_use": {"web_search_requests": 1},
            }
        }
    )

    assert usage["server_tool_use"] == {"web_search_requests": 1}


def test_transport_wraps_successful_non_json_response(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = "not json"

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("bad json")

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(LLMRequestError, match="non-JSON"):
        make_deepseek_anthropic_messages_request(
            {"model": "deepseek-v4-pro", "messages": [], "tools": [], "max_tokens": 1},
            "https://api.deepseek.com",
            "sk-test",
        )
