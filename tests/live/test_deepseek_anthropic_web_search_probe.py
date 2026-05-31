"""Opt-in DeepSeek Anthropic hosted web-search probes."""

from __future__ import annotations

import os

import pytest

from llm_router.deepseek.anthropic_web_search import (
    make_deepseek_anthropic_messages_request,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_ROUTER_LIVE_DEEPSEEK_WEB_SEARCH") != "1",
    reason="set LLM_ROUTER_LIVE_DEEPSEEK_WEB_SEARCH=1 to run live DeepSeek probes",
)


def test_live_deepseek_web_search_20250305_shape():
    api_key = os.environ["DEEPSEEK_API_KEY"]
    payload = {
        "model": os.environ.get("DEEPSEEK_WEB_SEARCH_MODEL", "deepseek-v4-pro"),
        "max_tokens": 512,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": "What is the latest OpenAI platform changelog title? Answer briefly.",
            }
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }

    response = make_deepseek_anthropic_messages_request(
        payload,
        "https://api.deepseek.com",
        api_key,
    )

    content = response.get("content") or []
    content_types = [block.get("type") for block in content if isinstance(block, dict)]
    assert "server_tool_use" in content_types
    assert "web_search_tool_result" in content_types
    assert response.get("usage", {}).get("server_tool_use", {}).get("web_search_requests", 0) >= 1


def test_live_deepseek_web_search_20260209_probe_only():
    api_key = os.environ["DEEPSEEK_API_KEY"]
    payload = {
        "model": os.environ.get("DEEPSEEK_WEB_SEARCH_MODEL", "deepseek-v4-pro"),
        "max_tokens": 128,
        "temperature": 0,
        "messages": [{"role": "user", "content": "Search for OpenAI."}],
        "tools": [{"type": "web_search_20260209", "name": "web_search"}],
    }

    response = make_deepseek_anthropic_messages_request(
        payload,
        "https://api.deepseek.com",
        api_key,
    )

    assert "content" in response
