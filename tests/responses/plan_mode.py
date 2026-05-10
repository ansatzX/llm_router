"""Responses endpoint regression tests."""

import llm_router.server as server_mod
from llm_router.config import UpstreamConfig
from tests.responses._helpers import _configure_test_app


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
    server_mod._config.default_model_type = "responses_chat"
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
    server_mod._config.default_model_type = "responses_chat"
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
    server_mod._config.default_model_type = "responses_chat"
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
    server_mod._config.default_model_type = "responses_chat"
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
    server_mod._config.default_model_type = "responses_chat"
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
