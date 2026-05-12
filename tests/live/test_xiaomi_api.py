"""Opt-in live Xiaomi MiMo provider smoke tests.

These tests call Xiaomi's official Chat Completions API through the local
adapter boundary. They are skipped by default because they require credentials
and consume provider quota.
"""

from __future__ import annotations

import os

import pytest

from llm_router.llm_client import make_llm_request
from llm_router.xiaomi import XiaomiChatAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_ROUTER_LIVE_XIAOMI") != "1",
    reason="set LLM_ROUTER_LIVE_XIAOMI=1 to run live Xiaomi provider tests",
)


BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com")
MODEL = os.environ.get("MIMO_LIVE_MODEL", "mimo-v2.5-pro")
IMAGE_URL = (
    "https://example-files.cnbj1.mi-fds.com/example-files/image/image_example.png"
)


def _api_key() -> str:
    key = os.environ.get("MIMO_API_KEY")
    if not key:
        pytest.skip("MIMO_API_KEY is required for live Xiaomi provider tests")
    return key


def _request(payload: dict) -> dict:
    adapter = XiaomiChatAdapter()
    return make_llm_request(
        adapter.filter_request_payload(payload),
        BASE_URL,
        _api_key(),
    )


def test_live_xiaomi_basic_text_completion():
    response = _request({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say exactly: router-ok"}],
        "max_completion_tokens": 64,
        "thinking": {"type": "disabled"},
        "stream": False,
    })

    assert "router-ok" in response["choices"][0]["message"]["content"]


def test_live_xiaomi_image_url_completion():
    response = _request({
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "text", "text": "Describe the image in one sentence."},
                ],
            }
        ],
        "max_completion_tokens": 128,
        "thinking": {"type": "disabled"},
        "stream": False,
    })

    assert response["choices"][0]["message"]["content"]


def test_live_xiaomi_thinking_with_function_tool_call():
    response = _request({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Call echo with value router-ok."}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_completion_tokens": 256,
        "thinking": {"type": "enabled"},
        "stream": False,
    })

    message = response["choices"][0]["message"]
    assert message.get("content") or message.get("tool_calls")
    if message.get("tool_calls"):
        assert "reasoning_content" in message


def test_live_xiaomi_web_search_tool_smoke():
    response = _request({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Search for today's Xiaomi MiMo news."}],
        "tools": [
            {
                "type": "web_search",
                "max_keyword": 1,
                "force_search": True,
                "limit": 1,
            }
        ],
        "tool_choice": "auto",
        "max_completion_tokens": 256,
        "thinking": {"type": "disabled"},
        "stream": False,
    })

    message = response["choices"][0]["message"]
    assert message.get("content")
    assert message.get("annotations")
