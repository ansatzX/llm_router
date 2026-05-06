"""Tests for DeepSeek provider-specific Chat adapter."""

from llm_router.deepseek import DeepSeekChatAdapter


def test_deepseek_filters_responses_metadata_from_chat_payload():
    adapter = DeepSeekChatAdapter()

    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "reasoning": None,
        "prompt_cache_key": "cache-key",
        "client_metadata": {"x-codex-installation-id": "install-id"},
        "text": {"format": {"type": "json_schema"}},
    }

    filtered = adapter.filter_request_payload(payload)

    assert filtered == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 0.7,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    assert payload["client_metadata"] == {"x-codex-installation-id": "install-id"}


def test_deepseek_maps_responses_reasoning_effort_and_drops_service_tier():
    adapter = DeepSeekChatAdapter()

    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "reasoning": {"effort": "medium", "summary": "detailed"},
        "service_tier": "priority",
    }

    filtered = adapter.filter_request_payload(payload)

    assert filtered == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "reasoning_effort": "medium",
    }


def test_deepseek_gateway_can_forward_compat_prompt_cache_fields():
    adapter = DeepSeekChatAdapter(forward_compat_prompt_cache=True)

    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "prompt_cache_key": "cache-key",
        "prompt_cache_retention": {"type": "persistent"},
        "reasoning": {"effort": "high"},
    }

    filtered = adapter.filter_request_payload(payload)

    assert filtered == {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "prompt_cache_key": "cache-key",
        "prompt_cache_retention": {"type": "persistent"},
        "reasoning_effort": "high",
    }


def test_deepseek_groups_parallel_response_function_calls():
    adapter = DeepSeekChatAdapter()

    messages = adapter.flatten_response_items(
        [
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_1",
                "reasoning_content": "inspect repo",
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"git status"}',
                "call_id": "call_2",
                "reasoning_content": "inspect repo",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ls output",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "status output",
            },
        ]
    )

    assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
    assert messages[0]["reasoning_content"] == "inspect repo"
    assert [tc["id"] for tc in messages[0]["tool_calls"]] == [
        "call_1",
        "call_2",
    ]


def test_deepseek_keeps_tool_outputs_adjacent_when_system_message_intervenes():
    adapter = DeepSeekChatAdapter()

    messages = adapter.flatten_response_items(
        [
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"curl"}',
                "call_id": "call_1",
                "reasoning_content": "fetch issues",
            },
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Approved command prefix saved.",
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "curl output",
            },
        ]
    )

    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
        "system",
    ]
    assert messages[0]["tool_calls"][0]["id"] == "call_1"
    assert messages[1]["tool_call_id"] == "call_1"
    assert messages[2]["content"] == "Approved command prefix saved."


def test_deepseek_wraps_responses_only_tools_as_chat_functions():
    adapter = DeepSeekChatAdapter()

    tools = adapter.responses_tools_to_chat(
        [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "freeform patch",
            },
        ]
    )

    assert tools[0]["function"]["name"] == "apply_patch"
    assert tools[0]["function"]["parameters"]["required"] == ["input"]


def test_deepseek_round_trips_tool_reasoning_from_cache():
    adapter = DeepSeekChatAdapter()
    adapter.record_tool_reasoning("call_1", "tool reasoning")

    messages = adapter.flatten_response_items(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_1",
            },
        ]
    )

    assert "reasoning_content" not in messages[0]
    assert messages[1]["reasoning_content"] == "tool reasoning"


def test_deepseek_hydrates_tool_reasoning_cache_from_provider_state():
    adapter = DeepSeekChatAdapter()
    adapter.load_provider_state(
        {
            "reasoning_by_call_id": {"call_1": "persisted tool reasoning"},
        }
    )

    messages = adapter.flatten_response_items(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_1",
            },
        ]
    )

    assert "reasoning_content" not in messages[0]
    assert messages[1]["reasoning_content"] == "persisted tool reasoning"


def test_deepseek_exports_reasoning_cache_as_provider_state():
    adapter = DeepSeekChatAdapter()
    adapter.record_tool_reasoning("call_1", "tool reasoning")

    assert adapter.dump_provider_state() == {
        "reasoning_by_call_id": {"call_1": "tool reasoning"},
    }


def test_deepseek_replays_uncached_tool_turns_without_text_transcript():
    """Missing reasoning sidecars must not be downgraded to user-visible text."""
    adapter = DeepSeekChatAdapter()

    messages = adapter.flatten_response_items(
        [
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_missing_reasoning",
            },
            {
                "type": "function_call_output",
                "call_id": "call_missing_reasoning",
                "output": "README.md\n",
            },
        ]
    )

    assert [message["role"] for message in messages] == [
        "assistant",
        "tool",
    ]
    assert messages[0]["tool_calls"][0]["id"] == "call_missing_reasoning"
    assert messages[1]["tool_call_id"] == "call_missing_reasoning"
    assert "historical tool call omitted" not in str(messages)
    assert "Tool output:" not in str(messages)


def test_deepseek_restores_chat_response_tool_calls_to_responses_items():
    adapter = DeepSeekChatAdapter()

    output_items, output_text, tool_calls = adapter.chat_response_to_output_items(
        {
            "content": "",
            "reasoning_content": "edit the file",
            "tool_calls": [
                {
                    "id": "call_patch",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": '{"input":"*** Begin Patch\\n*** End Patch\\n"}',
                    },
                }
            ],
        },
        {"apply_patch": "custom"},
    )

    assert output_text is None
    assert tool_calls[0]["id"] == "call_patch"
    assert output_items == [
        {
            "type": "custom_tool_call",
            "id": "call_patch",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch\n",
            "reasoning_content": "edit the file",
        }
    ]
    assert adapter.reasoning_by_call_id["call_patch"] == "edit the file"


def test_deepseek_restores_plain_chat_response_with_inline_reasoning_only():
    adapter = DeepSeekChatAdapter()

    output_items, output_text, tool_calls = adapter.chat_response_to_output_items(
        {
            "content": "done",
            "reasoning_content": "plain response reasoning",
        },
        {},
    )

    assert output_text == "done"
    assert tool_calls == []
    assert output_items == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
            "reasoning_content": "plain response reasoning",
        }
    ]
    assert adapter.dump_provider_state() == {"reasoning_by_call_id": {}}
