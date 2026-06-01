"""Responses endpoint regression tests."""

import json
from types import SimpleNamespace

import llm_router.server as server_mod
from llm_router.config import UpstreamConfig
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
    server_mod._config.default_model_type = "responses_chat"

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
    mock_make_request.side_effect = [
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "found it",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/weather",
                                "title": "Weather",
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {
            "created": 123,
            "choices": [{"message": {"content": "final"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    server_mod._config.default_model_type = "responses_chat"

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

def test_responses_deepseek_ignores_multimodal_input_and_keeps_text(
    tmp_path,
    monkeypatch,
):
    """DeepSeek policy degrades unsupported images instead of failing the turn."""
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

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe this"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["messages"] == [{"role": "user", "content": "describe this"}]

def test_responses_deepseek_filters_hosted_web_search_tool_before_upstream(
    tmp_path,
    monkeypatch,
):
    """DeepSeek policy returns a normal empty-result turn without forwarding hosted tools."""
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
    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.DeepSeekAnthropicWebSearchBridge.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Inactive web_search must stay on the Chat adapter path"),
        ),
    )

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "search the web",
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert "tools" not in payload


def test_responses_deepseek_hosted_web_search_uses_anthropic_bridge(
    tmp_path,
    monkeypatch,
):
    """Official DeepSeek hosted web_search should use the whole-turn bridge."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    bridge_calls: list[dict[str, object]] = []

    def fake_bridge_run(
        self,
        *,
        messages,
        tools_raw,
        max_tokens=None,
        temperature=None,
        top_p=None,
        request_options=None,
    ):
        bridge_calls.append({
            "base_url": self.base_url,
            "model": self.model,
            "messages": messages,
            "tools_raw": tools_raw,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "request_options": request_options,
        })
        return SimpleNamespace(
            output_items=[
                {
                    "type": "web_search_call",
                    "id": "srvtoolu_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "Search current docs"},
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Found the latest docs."}],
                },
            ],
            output_text="Found the latest docs.",
            usage={
                "input_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "server_tool_use": {"web_search_requests": 1},
                "total_tokens": 5,
            },
            raw_response={
                "created": 124,
                "content": [{"type": "text", "text": "Found the latest docs."}],
            },
        )

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.DeepSeekAnthropicWebSearchBridge.run",
        fake_bridge_run,
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
            "input": "Search current docs",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    assert bridge_calls == [
        {
            "base_url": "https://api.deepseek.com",
            "model": "test-model",
            "messages": [{"role": "user", "content": "Search current docs"}],
            "tools_raw": [{"type": "web_search", "external_web_access": True}],
            "max_tokens": None,
            "temperature": None,
            "top_p": None,
            "request_options": {},
        }
    ]
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == ["web_search_call", "message"]
    assert body["output"][0]["id"] == "srvtoolu_1"
    assert body["output"][0]["action"] == {"type": "search", "query": "Search current docs"}
    assert body["output"][1]["content"][0]["text"] == "Found the latest docs."
    assert body["output_text"] == "Found the latest docs."
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}


def test_responses_deepseek_web_search_bridge_recovers_dsml_follow_up_queries(
    tmp_path,
    monkeypatch,
):
    """Provider DSML web_search residuals should be converted into hosted searches."""
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
                    {
                        "type": "server_tool_use",
                        "id": "srv_1",
                        "name": "web_search",
                        "input": {"query": "Cunxi Gong"},
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
                            '<｜｜DSML｜｜parameter name="query" string="true">'
                            "Cunxi Gong researcher affiliation"
                            "</｜｜DSML｜｜parameter>"
                            "</｜｜DSML｜｜invoke>"
                            "</｜｜DSML｜｜tool_calls>"
                        ),
                    },
                ],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
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
                    "input": {"query": "Cunxi Gong researcher affiliation"},
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srv_2",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://example.com/cunxi-gong",
                            "title": "Cunxi Gong",
                        }
                    ],
                },
                {"type": "text", "text": "Recovered answer."},
            ],
            "usage": {
                "input_tokens": 4,
                "output_tokens": 5,
                "server_tool_use": {"web_search_requests": 1},
            },
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

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "搜索一下，谁是Cunxi Gong",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == [
        "web_search_call",
        "web_search_call",
        "message",
    ]
    assert body["output"][1]["action"] == {
        "type": "search",
        "query": "Cunxi Gong researcher affiliation",
    }
    assert body["output_text"] == "Recovered answer."
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 2}


def test_responses_deepseek_web_search_bridge_preserves_function_tool_calls(
    tmp_path,
    monkeypatch,
):
    """Mixed hosted web_search and ordinary tools must keep tool-call state."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    function_call = {
        "type": "function_call",
        "id": "toolu_1",
        "call_id": "toolu_1",
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }

    def fake_bridge_run(
        self,
        *,
        messages,
        tools_raw,
        max_tokens=None,
        temperature=None,
        top_p=None,
        request_options=None,
    ):
        return SimpleNamespace(
            output_items=[function_call],
            output_text=None,
            usage={
                "input_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "server_tool_use": {"web_search_requests": 0},
                "total_tokens": 3,
            },
            raw_response={"created": 125, "stop_reason": "tool_use"},
        )

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.DeepSeekAnthropicWebSearchBridge.run",
        fake_bridge_run,
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
            "input": "Search docs, then inspect README if needed.",
            "tools": [
                {"type": "web_search", "external_web_access": True},
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            ],
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    body = response.get_json()
    assert body["output"] == [function_call]
    assert body["output_text"] is None
    assert body["tool_calls"] == [function_call]
    assert len(server_mod._sessions) == 1


def test_responses_deepseek_web_search_bridge_forwards_request_controls(
    tmp_path,
    monkeypatch,
):
    """Bridge path must preserve supported Codex request-control fields."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    bridge_calls: list[dict[str, object]] = []

    def fake_bridge_run(
        self,
        *,
        messages,
        tools_raw,
        max_tokens=None,
        temperature=None,
        top_p=None,
        request_options=None,
    ):
        bridge_calls.append({
            "request_options": request_options,
            "max_tokens": max_tokens,
        })
        return SimpleNamespace(
            output_items=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "No search used."}],
                }
            ],
            output_text="No search used.",
            usage={
                "input_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 3,
                "output_tokens_details": {"reasoning_tokens": 0},
                "server_tool_use": {"web_search_requests": 0},
                "total_tokens": 5,
            },
            raw_response={"created": 126, "stop_reason": "end_turn"},
        )

    monkeypatch.setattr(
        "llm_router.deepseek.anthropic_web_search.DeepSeekAnthropicWebSearchBridge.run",
        fake_bridge_run,
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
            "input": "Search current docs",
            "tools": [{"type": "web_search", "external_web_access": True}],
            "tool_choice": "none",
            "reasoning": {"effort": "high"},
            "stop": "DONE",
            "max_output_tokens": 4096,
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 0
    assert bridge_calls == [
        {
            "request_options": {
                "tool_choice": {"type": "none"},
                "output_config": {"effort": "high"},
                "stop_sequences": ["DONE"],
            },
            "max_tokens": 4096,
        }
    ]


def test_responses_deepseek_mcp_first_web_search_stays_on_chat_path(
    tmp_path,
    monkeypatch,
):
    """DeepSeek mcp_first routes should not receive the new internal search tool."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_request.return_value = {
        "created": 123,
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    server_mod._config.default_model_type = "mcp_first"
    server_mod._config.upstreams["deepseek"] = UpstreamConfig(
        base_url="https://api.deepseek.com",
    )
    server_mod._config.default_upstream = "deepseek"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "Search current docs",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 200
    mock_make_request.assert_called_once()
    payload = mock_make_request.call_args.args[0]
    assert "tools" not in payload

def test_responses_xiaomi_exposes_do_web_search_without_eager_search(
    tmp_path,
    monkeypatch,
):
    """Hosted Xiaomi web_search becomes a model-chosen internal function."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [
                {"message": {"content": "no search needed"}, "finish_reason": "stop"}
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
            "input": "Improve documentation in @filename",
            "tools": [
                {
                    "type": "web_search",
                    "external_web_access": True,
                    "force_search": True,
                    "max_keyword": 3,
                    "limit": 1,
                    "user_location": {
                        "type": "approximate",
                        "country": "China",
                        "region": "Hubei",
                        "city": "Wuhan",
                    },
                }
            ],
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 1
    main_payload = mock_make_request.call_args.args[0]
    assert main_payload["thinking"] == {"type": "enabled"}
    assert main_payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "do_web_search",
                "description": (
                    "Search the live web when current or external information is needed. "
                    "Call this tool only when the user request requires web results. "
                    "The router will execute Xiaomi MiMo web_search and return text or null."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Focused web search query.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                    "x-xiaomi-web-search-defaults": {
                        "max_keyword": 3,
                        "limit": 1,
                        "force_search": True,
                        "user_location": {
                            "type": "approximate",
                            "country": "China",
                            "region": "Hubei",
                            "city": "Wuhan",
                        },
                    },
                },
            },
        }
    ]
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == ["message"]
    assert body["output"][0]["content"][0]["text"] == "no search needed"


def test_responses_xiaomi_runs_do_web_search_and_continues_main_request(
    tmp_path,
    monkeypatch,
):
    """If Xiaomi chooses do_web_search, router executes search then asks again."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    mock_make_request.side_effect = [
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "do_web_search",
                                    "arguments": '{"query":"Wuhan weather tomorrow"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        },
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "Wuhan tomorrow weather from search",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/weather",
                                "title": "Weather",
                                "summary": "Wuhan weather summary",
                            },
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
        {
            "created": 123,
            "choices": [
                {"message": {"content": "final with search"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 9, "total_tokens": 17},
        },
    ]
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://api.xiaomimimo.com",
    )
    server_mod._config.default_upstream = "xiaomi"

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "input": "武汉明天天气怎么样？",
            "tools": [{"type": "web_search", "external_web_access": True}],
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 3
    first_payload = mock_make_request.call_args_list[0].args[0]
    search_payload = mock_make_request.call_args_list[1].args[0]
    final_payload = mock_make_request.call_args_list[2].args[0]
    assert first_payload["tools"][0]["function"]["name"] == "do_web_search"
    assert search_payload["model"] == "mimo-v2-omni"
    assert search_payload["thinking"] == {"type": "disabled"}
    assert search_payload["tools"] == [
        {"type": "web_search", "force_search": True, "max_keyword": 3, "limit": 1}
    ]
    assert final_payload["messages"][-2]["tool_calls"][0]["function"]["name"] == (
        "do_web_search"
    )
    assert final_payload["messages"][-1]["role"] == "tool"
    assert final_payload["messages"][-1]["tool_call_id"] == "call_search"
    assert "Wuhan tomorrow weather from search" in final_payload["messages"][-1]["content"]
    assert "https://example.com/weather" in final_payload["messages"][-1]["content"]
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == ["message"]
    assert body["output"][0]["content"][0]["text"] == "final with search"


def test_responses_xiaomi_do_web_search_failure_returns_null_tool_output(
    tmp_path,
    monkeypatch,
):
    """Search failure is debug-logged and returned as null to the main model."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    from llm_router.llm_client import LLMRequestError

    mock_make_request.side_effect = [
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_search",
                                "type": "function",
                                "function": {
                                    "name": "do_web_search",
                                    "arguments": '{"query":"news"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        LLMRequestError(
            "Param Incorrect",
            status_code=400,
            body={"error": {"message": "Param Incorrect"}},
        ),
        {
            "created": 123,
            "choices": [{"message": {"content": "final despite null"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    events = []
    monkeypatch.setattr(server_mod, "log_debug", lambda event, data: events.append((event, data)))
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://api.xiaomimimo.com",
    )
    server_mod._config.default_upstream = "xiaomi"

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "input": "Search current news",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 200
    final_payload = mock_make_request.call_args_list[2].args[0]
    assert final_payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_search",
        "content": "null",
    }
    assert any(
        event == "XIAOMI_BUILTIN_WEB_SEARCH_FAILED"
        and data["provider_status"] == 400
        and data["message"] == "Param Incorrect"
        for event, data in events
    )
    body = response.get_json()
    assert body["output"][0]["content"][0]["text"] == "final despite null"


def test_responses_xiaomi_questions_model_after_five_searches_then_stops(
    tmp_path,
    monkeypatch,
):
    """After five searches, ask the model whether it really needs more search."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    def tool_call_response(index: int) -> dict[str, object]:
        return {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_search_{index}",
                                "type": "function",
                                "function": {
                                    "name": "do_web_search",
                                    "arguments": f'{{"query":"query {index}"}}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def search_response(index: int) -> dict[str, object]:
        return {
            "created": 123,
            "choices": [
                {
                    "message": {"content": f"search result {index}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    side_effects: list[dict[str, object]] = [tool_call_response(1)]
    for index in range(1, 6):
        side_effects.append(search_response(index))
        side_effects.append(tool_call_response(index + 1))
    side_effects.append({
        "created": 123,
        "choices": [
            {
                "message": {"content": "I should stop and answer now."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    mock_make_request.side_effect = side_effects

    events = []
    monkeypatch.setattr(server_mod, "log_debug", lambda event, data: events.append((event, data)))
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://api.xiaomimimo.com",
    )
    server_mod._config.default_upstream = "xiaomi"

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "input": "持续搜索直到有结论",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 12
    question_payload = mock_make_request.call_args_list[11].args[0]
    assert question_payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_search_6",
        "content": json.dumps(
            "已经多次搜索了，是否继续？如果确实必须继续搜索，请再次调用 do_web_search；"
            "否则请基于已有搜索结果回答用户。",
            ensure_ascii=False,
        ),
    }
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == ["reasoning", "message"]
    assert body["output"][0]["summary"] == [
        {"type": "summary_text", "text": "正在多次搜索，提醒用户"}
    ]
    assert body["output"][0]["content"] == []
    assert body["output"][1]["content"][0]["text"] == "I should stop and answer now."
    assert any(
        event == "XIAOMI_INTERNAL_TOOL_LOOP_STOPPED"
        and data["reason"] == "questioned_model_after_max_search_rounds"
        and data["max_rounds"] == 5
        for event, data in events
    )


def test_responses_xiaomi_continues_when_model_insists_after_search_question(
    tmp_path,
    monkeypatch,
):
    """If the model asks again after the question, router resumes searching."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "unused"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )

    def tool_call_response(index: int) -> dict[str, object]:
        return {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_search_{index}",
                                "type": "function",
                                "function": {
                                    "name": "do_web_search",
                                    "arguments": f'{{"query":"query {index}"}}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def search_response(index: int) -> dict[str, object]:
        return {
            "created": 123,
            "choices": [
                {
                    "message": {"content": f"search result {index}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    side_effects: list[dict[str, object]] = [tool_call_response(1)]
    for index in range(1, 6):
        side_effects.append(search_response(index))
        side_effects.append(tool_call_response(index + 1))
    side_effects.extend([
        tool_call_response(7),
        search_response(7),
        {
            "created": 123,
            "choices": [
                {
                    "message": {
                        "content": "final after insisted search",
                        "reasoning_content": "provider reasoning after search",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ])
    mock_make_request.side_effect = side_effects

    events = []
    monkeypatch.setattr(server_mod, "log_debug", lambda event, data: events.append((event, data)))
    server_mod._config.default_model_type = "responses_chat"
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://api.xiaomimimo.com",
    )
    server_mod._config.default_upstream = "xiaomi"

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "input": "持续搜索直到有结论",
            "tools": [{"type": "web_search", "external_web_access": True}],
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 14
    search_payload = mock_make_request.call_args_list[12].args[0]
    assert search_payload["messages"] == [{"role": "user", "content": "query 7"}]
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == [
        "reasoning",
        "reasoning",
        "message",
    ]
    assert body["output"][0]["summary"] == [
        {"type": "summary_text", "text": "正在多次搜索，提醒用户"}
    ]
    assert body["output"][0]["content"] == []
    assert body["output"][1]["summary"][0]["text"].startswith("**")
    assert body["output"][1]["summary"][0]["text"] != "正在多次搜索，提醒用户"
    assert body["output"][1]["content"] == [
        {"type": "reasoning_text", "text": "provider reasoning after search"}
    ]
    assert body["output"][2]["content"][0]["text"] == "final after insisted search"
    assert any(
        event == "XIAOMI_INTERNAL_WEB_SEARCH_TOOL_CALLS"
        and data["round"] == 1
        and data["queries"] == ["query 7"]
        for event, data in events
    )


def test_responses_xiaomi_builtin_web_search_keeps_function_tools_on_main_request(
    tmp_path,
    monkeypatch,
):
    """Hosted web_search is replaced while normal function tools remain available."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "final"}, "finish_reason": "stop"}],
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
            "input": "Need current info",
            "tools": [
                {
                    "type": "function",
                    "name": "local_lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
                {"type": "web_search", "external_web_access": True},
            ],
            "reasoning": {"effort": "high"},
        },
    )

    assert response.status_code == 200
    main_payload = mock_make_request.call_args.args[0]
    assert [tool["function"]["name"] for tool in main_payload["tools"]] == [
        "local_lookup",
        "do_web_search",
    ]
    assert main_payload["tools"][0] == {
        "type": "function",
        "function": {
            "name": "local_lookup",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }
    body = response.get_json()
    assert [item["type"] for item in body["output"]] == ["message"]


def test_responses_xiaomi_maps_none_reasoning_to_disabled_thinking(
    tmp_path,
    monkeypatch,
):
    """Codex reasoning effort must control Xiaomi's documented thinking knob."""
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
    server_mod._config.upstreams["xiaomi"] = UpstreamConfig(
        base_url="https://api.xiaomimimo.com",
    )
    server_mod._config.default_upstream = "xiaomi"

    response = client.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-pro",
            "input": "hi",
            "reasoning": {"effort": "none"},
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning" not in payload
    assert "reasoning_effort" not in payload

def test_responses_xiaomi_stream_does_not_eagerly_run_web_search(
    tmp_path,
    monkeypatch,
):
    """Simulated SSE should contain only the model result if search is not chosen."""
    client, mock_make_request = _configure_test_app(
        tmp_path,
        monkeypatch,
        {
            "created": 123,
            "choices": [{"message": {"content": "plain answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    monkeypatch.setattr(
        server_mod,
        "make_llm_stream_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Xiaomi web_search should not use live upstream streaming"),
        ),
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
            "input": "search",
            "stream": True,
            "tools": [{"type": "web_search", "force_search": True}],
        },
    )

    assert response.status_code == 200
    assert mock_make_request.call_count == 1
    body = response.data.decode("utf-8")
    added = _sse_payloads(body, "response.output_item.added")
    done = _sse_payloads(body, "response.output_item.done")
    assert added[0]["item"]["type"] == "message"
    assert done[0]["item"]["type"] == "message"
    assert "web_search_call" not in body

def test_responses_generic_chat_does_not_apply_deepseek_unsupported_feature_policy(
    tmp_path,
    monkeypatch,
):
    """Unsupported-feature policy belongs to the selected provider adapter."""
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
    server_mod._config.default_upstream = "default"

    response = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe this"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                    ],
                }
            ],
            "tools": [{"type": "web_search", "external_web_access": False}],
        },
    )

    assert response.status_code == 200
    mock_make_request.assert_called_once()

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
    server_mod._config.default_model_type = "responses_chat"

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
    server_mod._config.default_model_type = "responses_chat"
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

def test_responses_chat_route_expands_namespace_tools_for_deepseek(
    tmp_path,
    monkeypatch,
):
    """Namespace children should be visible to DeepSeek as callable functions."""
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
            "input": "analyze image",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__Local_Read__",
                    "description": "Local Read tools.",
                    "tools": [
                        {
                            "type": "function",
                            "name": "analyze_image",
                            "description": "Analyze an image.",
                            "parameters": {
                                "type": "object",
                                "properties": {"image_path": {"type": "string"}},
                                "required": ["image_path"],
                            },
                        },
                        {
                            "type": "function",
                            "name": "process_binary_file",
                            "description": "Process a binary file.",
                            "parameters": {
                                "type": "object",
                                "properties": {"file_path": {"type": "string"}},
                                "required": ["file_path"],
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = mock_make_request.call_args.args[0]
    forwarded_names = [
        tool["function"]["name"]
        for tool in payload["tools"]
    ]
    assert forwarded_names == [
        "mcp__Local_Read__analyze_image",
        "mcp__Local_Read__process_binary_file",
    ]
    assert payload["tools"][0]["function"]["description"] == "Analyze an image."
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["image_path"]

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
    server_mod._config.default_model_type = "responses_chat"

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
    server_mod._config.default_model_type = "responses_chat"

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

def test_responses_chat_route_restores_namespace_tool_calls_as_response_items(
    tmp_path,
    monkeypatch,
):
    """Flattened DeepSeek calls should regain Codex namespace identity."""
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
                                "id": "call_image",
                                "type": "function",
                                "function": {
                                    "name": "mcp__Local_Read__analyze_image",
                                    "arguments": '{"image_path":"/tmp/input.png"}',
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
        json={
            "model": "test-model",
            "input": "analyze image",
            "tools": [
                {
                    "type": "namespace",
                    "name": "mcp__Local_Read__",
                    "tools": [
                        {
                            "type": "function",
                            "name": "analyze_image",
                            "parameters": {"type": "object"},
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["output"] == [
        {
            "type": "function_call",
            "id": "call_image",
            "call_id": "call_image",
            "namespace": "mcp__Local_Read__",
            "name": "analyze_image",
            "arguments": '{"image_path":"/tmp/input.png"}',
        }
    ]

def test_responses_chat_route_replays_namespace_tool_calls_flattened_for_deepseek(
    tmp_path,
    monkeypatch,
):
    """Committed namespace tool calls must replay with DeepSeek-visible names."""
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
                                "id": "call_image",
                                "type": "function",
                                "function": {
                                    "name": "mcp__Local_Read__analyze_image",
                                    "arguments": '{"image_path":"/tmp/input.png"}',
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
    tools = [
        {
            "type": "namespace",
            "name": "mcp__Local_Read__",
            "tools": [
                {
                    "type": "function",
                    "name": "analyze_image",
                    "parameters": {"type": "object"},
                },
            ],
        }
    ]

    first = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "analyze image", "tools": tools},
    )
    first_body = first.get_json()

    mock_make_request.reset_mock()
    mock_make_request.return_value = {
        "created": 124,
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    second = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "previous_response_id": first_body["id"],
            "tools": tools,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_image",
                    "output": "image summary",
                }
            ],
        },
    )

    assert second.status_code == 200
    payload = mock_make_request.call_args.args[0]
    assistant_tool_messages = [
        message for message in payload["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert assistant_tool_messages[0]["tool_calls"][0]["function"]["name"] == (
        "mcp__Local_Read__analyze_image"
    )

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
    server_mod._config.default_model_type = "responses_chat"

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
    added = _sse_payloads(event_stream, "response.output_item.added")
    deltas = _sse_payloads(event_stream, "response.custom_tool_call_input.delta")
    assert added[0]["item"]["type"] == "custom_tool_call"
    assert added[0]["item"]["input"] == ""
    assert deltas[0]["delta"] == "*** Begin Patch\n*** End Patch\n"
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
    server_mod._config.default_model_type = "responses_chat"

    response = client.post(
        "/v1/responses",
        json={"model": "test-model", "input": "list files"},
    )

    assert response.status_code == 200
    body = response.get_json()
    reasoning = body["output"][0]
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"][0]["type"] == "summary_text"
    assert reasoning["summary"][0]["text"] == ""
    assert reasoning["content"] == [
        {"type": "reasoning_text", "text": "I should list files first."},
    ]
    assert body["output"][1]["reasoning_content"] == "I should list files first."

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
