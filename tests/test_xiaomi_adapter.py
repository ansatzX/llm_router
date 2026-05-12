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


def test_xiaomi_web_search_does_not_downgrade_reasoning_by_default():
    adapter = XiaomiChatAdapter()

    filtered = adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "tools": [{"type": "web_search", "force_search": True}],
        "reasoning": {"effort": "high"},
    })

    assert filtered["thinking"] == {"type": "enabled"}


def test_xiaomi_preserves_explicit_thinking_over_reasoning_mapping():
    adapter = XiaomiChatAdapter()

    filtered = adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "thinking": {"type": "enabled"},
        "reasoning": {"effort": "none"},
    })

    assert filtered["thinking"] == {"type": "enabled"}


def test_xiaomi_preserves_explicit_thinking_for_web_search():
    adapter = XiaomiChatAdapter()

    filtered = adapter.filter_request_payload({
        "model": "mimo-v2.5-pro",
        "messages": [],
        "tools": [{"type": "web_search", "force_search": True}],
        "thinking": {"type": "enabled"},
        "reasoning": {"effort": "high"},
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


def test_xiaomi_adds_defaults_for_bare_web_search_tool():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat([{"type": "web_search"}])

    assert tools == [
        {
            "type": "web_search",
            "force_search": True,
            "max_keyword": 3,
            "limit": 1,
        }
    ]


def test_xiaomi_accepts_provider_forced_search_alias_as_force_search():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat([
        {"type": "web_search", "forced_search": False},
    ])

    assert tools == [
        {
            "type": "web_search",
            "force_search": False,
            "max_keyword": 3,
            "limit": 1,
        }
    ]


def test_xiaomi_sanitizes_nullable_function_schema_for_chat_tools():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat([
        {
            "type": "function",
            "name": "search_issues",
            "description": "search issues",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "null"},
                        ],
                        "description": "repository name",
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "maximum results",
                    },
                    "query": {"type": "string"},
                },
                "required": ["repo", "limit", "query"],
                "additionalProperties": False,
            },
        },
    ])

    parameters = tools[0]["function"]["parameters"]
    assert parameters == {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "repository name",
            },
            "limit": {
                "type": "integer",
                "description": "maximum results",
            },
            "query": {"type": "string"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    assert "anyOf" not in json.dumps(parameters)
    assert '"null"' not in json.dumps(parameters)


def test_xiaomi_sanitizes_multi_shape_function_schema_for_chat_tools():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat([
        {
            "type": "function",
            "name": "github_search",
            "description": "search files",
            "parameters": {
                "type": "object",
                "properties": {
                    "repository_name": {
                        "description": "Repository or repositories to search within.",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                    },
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    ])

    parameters = tools[0]["function"]["parameters"]
    assert parameters["properties"]["repository_name"] == {
        "description": "Repository or repositories to search within.",
        "type": "string",
    }
    assert parameters["additionalProperties"] is False
    serialized = json.dumps(parameters)
    assert "anyOf" not in serialized
    assert '"null"' not in serialized


def test_xiaomi_adds_additional_properties_false_to_nested_objects():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat([
        {
            "type": "function",
            "name": "review",
            "parameters": {
                "type": "object",
                "properties": {
                    "comments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "body": {"type": "string"},
                            },
                            "required": ["path", "body"],
                        },
                    },
                },
            },
        },
    ])

    parameters = tools[0]["function"]["parameters"]
    nested = parameters["properties"]["comments"]["items"]
    assert parameters["additionalProperties"] is False
    assert nested["additionalProperties"] is False


def test_xiaomi_replaces_additional_properties_true_for_arbitrary_objects():
    adapter = XiaomiChatAdapter()

    tools = adapter.responses_tools_to_chat([
        {
            "type": "function",
            "name": "create_tree",
            "parameters": {
                "type": "object",
                "properties": {
                    "tree_elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        },
                    },
                },
            },
        },
    ])

    items_schema = tools[0]["function"]["parameters"]["properties"][
        "tree_elements"
    ]["items"]
    assert items_schema["additionalProperties"] is False


def test_xiaomi_tool_conversion_logs_payload_diagnostics(monkeypatch):
    adapter = XiaomiChatAdapter()
    events = []

    monkeypatch.setattr(
        "llm_router.xiaomi.chat.log_debug",
        lambda event, data: events.append((event, data)),
    )

    adapter.responses_tools_to_chat([
        {
            "type": "function",
            "name": "maybe_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                },
                "required": ["repo"],
            },
        },
        {"type": "web_search", "external_web_access": True},
    ])

    event, data = events[-1]
    assert event == "XIAOMI_CHAT_TOOL_DIAGNOSTICS"
    assert data["input_tool_count"] == 2
    assert data["forwarded_tool_count"] == 2
    assert data["function_count"] == 1
    assert data["web_search_count"] == 1
    assert data["schemas_with_any_of_before"] == 1
    assert data["schemas_with_any_of_after"] == 0
    assert data["schemas_with_null_type_after"] == 0
    assert data["object_schemas_missing_additional_properties_after"] == 0
    assert data["object_schemas_with_additional_properties_true_after"] == 0


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
