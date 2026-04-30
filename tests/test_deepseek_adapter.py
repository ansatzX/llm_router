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
            {"type": "web_search", "external_web_access": False},
        ]
    )

    assert tools[0]["function"]["name"] == "apply_patch"
    assert tools[0]["function"]["parameters"]["required"] == ["input"]
    assert tools[1]["function"]["name"] == "web_search"


def test_deepseek_round_trips_reasoning_from_cache():
    adapter = DeepSeekChatAdapter()
    adapter.record_message_reasoning("hello", "message reasoning")
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

    assert messages[0]["reasoning_content"] == "message reasoning"
    assert messages[1]["reasoning_content"] == "tool reasoning"


def test_deepseek_downgrades_uncached_tool_turns_without_reasoning():
    """Restarted router cannot safely replay DeepSeek thinking tool turns."""
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

    assert messages == [
        {
            "role": "user",
            "content": (
                "[historical tool call omitted: exec_command "
                "call_id=call_missing_reasoning]\n"
                '{"cmd":"ls"}\n'
                "Tool output:\nREADME.md\n"
            ),
        }
    ]


def test_deepseek_keeps_later_reasoned_tool_turns_after_legacy_turn():
    adapter = DeepSeekChatAdapter()

    messages = adapter.flatten_response_items(
        [
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"cat skill"}',
                "call_id": "call_without_reasoning",
            },
            {
                "type": "function_call_output",
                "call_id": "call_without_reasoning",
                "output": "skill text",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_with_reasoning",
                "reasoning_content": "inspect repo",
            },
            {
                "type": "function_call_output",
                "call_id": "call_with_reasoning",
                "output": "README.md",
            },
        ]
    )

    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "assistant",
        "tool",
    ]
    assert messages[0]["content"].startswith(
        "[historical tool call omitted: exec_command "
        "call_id=call_without_reasoning]"
    )
    assert messages[2]["reasoning_content"] == "inspect repo"
    assert messages[2]["tool_calls"][0]["id"] == "call_with_reasoning"


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


def test_deepseek_restores_plain_chat_response_and_caches_reasoning():
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
    assert (
        adapter.reasoning_by_message_content["done"]
        == "plain response reasoning"
    )
