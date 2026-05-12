"""Tests for Xiaomi MiMo provider-specific Chat adapter."""

import json

from llm_router.xiaomi import XiaomiChatAdapter


def test_xiaomi_filters_payload_to_official_chat_params():
    adapter = XiaomiChatAdapter()
    payload = {
        "model": "mimo-v2.5-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_completion_tokens": 1024,
        "thinking": {"type": "enabled"},
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "x"}}],
        "previous_response_id": "resp_1",
        "store": True,
        "include": ["reasoning.encrypted_content"],
        "client_metadata": {"installation": "test"},
        "service_tier": "priority",
    }

    filtered = adapter.filter_request_payload(payload)

    assert filtered == {
        "model": "mimo-v2.5-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_completion_tokens": 1024,
        "thinking": {"type": "enabled"},
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "x"}}],
    }


def test_xiaomi_drops_non_auto_tool_choice():
    adapter = XiaomiChatAdapter()
    payload = {
        "model": "mimo-v2.5-pro",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "tool_choice": "none",
    }

    filtered = adapter.filter_request_payload(payload)

    assert "tool_choice" not in filtered


def test_xiaomi_maps_codex_reasoning_to_thinking():
    adapter = XiaomiChatAdapter()

    assert adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "reasoning": {"effort": "none"},
    })["thinking"] == {"type": "disabled"}
    assert adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "reasoning_effort": "minimal",
    })["thinking"] == {"type": "disabled"}
    assert adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "reasoning": {"effort": "high"},
    })["thinking"] == {"type": "enabled"}


def test_xiaomi_preserves_explicit_thinking_over_reasoning_mapping():
    adapter = XiaomiChatAdapter()

    filtered = adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "thinking": {"type": "enabled"},
        "reasoning": {"effort": "none"},
    })

    assert filtered["thinking"] == {"type": "enabled"}


def test_xiaomi_preserves_developer_role():
    adapter = XiaomiChatAdapter()

    messages = adapter.flatten_response_items(
        [
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
    )

    assert messages[0] == {"role": "developer", "content": "follow policy"}
    assert messages[1] == {"role": "user", "content": "hello"}


def test_xiaomi_preserves_multimodal_content_parts():
    adapter = XiaomiChatAdapter()

    messages = adapter.flatten_response_items(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe this"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc",
                    },
                ],
            },
        ],
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        },
    ]


def test_xiaomi_preserves_structured_tool_output_images():
    adapter = XiaomiChatAdapter()

    messages = adapter.flatten_response_items(
        [
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": "tool screenshot"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc",
                    },
                ],
            },
        ],
    )

    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_image",
            "content": [
                {"type": "text", "text": "tool screenshot"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        },
    ]


def test_xiaomi_forwards_supported_web_search_tool():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat(
        [
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
                    "timezone": "Asia/Shanghai",
                },
                "search_context_size": "high",
                "filters": {"allowed_domains": ["example.com"]},
            },
        ],
    )

    assert tools == [
        {
            "type": "web_search",
            "force_search": True,
            "max_keyword": 3,
            "limit": 1,
            "user_location": {
                "type": "approximate",
                "country": "China",
                "region": "Hubei",
                "city": "Wuhan",
            },
        },
    ]


def test_xiaomi_skips_cached_web_search_tool():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat(
        [{"type": "web_search", "external_web_access": False}],
    )

    assert tools == []


def test_xiaomi_restores_reasoning_and_custom_tool_calls():
    adapter = XiaomiChatAdapter()

    output_items, output_text, native_tool_calls = adapter.chat_response_to_output_items(
        {
            "content": "",
            "reasoning_content": "need patch",
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
    assert native_tool_calls[0]["id"] == "call_patch"
    assert output_items[0]["type"] == "reasoning"
    assert output_items[1] == {
        "type": "custom_tool_call",
        "id": "call_patch",
        "call_id": "call_patch",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch\n",
        "reasoning_content": "need patch",
    }
    assert adapter.dump_provider_state() == {
        "reasoning_by_call_id": {"call_patch": "need patch"},
    }


def test_xiaomi_replays_custom_tool_calls_with_json_arguments():
    adapter = XiaomiChatAdapter()

    messages = adapter.flatten_response_items([
        {
            "type": "custom_tool_call",
            "id": "call_patch",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "--- a/file\n+++ b/file\n",
        },
    ])

    arguments = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"input": "--- a/file\n+++ b/file\n"}


def test_xiaomi_preserves_response_annotations_on_text_item():
    adapter = XiaomiChatAdapter()

    output_items, _, _ = adapter.chat_response_to_output_items(
        {
            "content": "answer with citation",
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.com/source",
                    "title": "Source",
                },
            ],
        },
        {},
    )

    assert output_items == [
        {
            "type": "web_search_call",
            "status": "completed",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "answer with citation",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://example.com/source",
                            "title": "Source",
                        },
                    ],
                },
            ],
        },
    ]


def test_xiaomi_replays_tool_reasoning_from_provider_state():
    adapter = XiaomiChatAdapter()
    adapter.load_provider_state(
        {"reasoning_by_call_id": {"call_1": "persisted reasoning"}},
    )

    messages = adapter.flatten_response_items(
        [
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_1",
            },
        ],
    )

    assert messages[0]["role"] == "assistant"
    assert messages[0]["reasoning_content"] == "persisted reasoning"
