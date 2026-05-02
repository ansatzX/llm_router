"""Focused tests for parser base data structures."""

from dataclasses import FrozenInstanceError

import pytest

from llm_router.parser.base import ParseResult, ToolCall


def test_tool_call_to_openai_format_serializes_nested_and_unicode_arguments():
    tool_call = ToolCall(
        tool_name="search",
        arguments={
            "query": "O'Brien",
            "filters": {"tags": ["python", "你好世界"]},
        },
    )

    openai_format = tool_call.to_openai_format()

    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "search"
    assert '"filters"' in openai_format["function"]["arguments"]
    assert "你好世界" in openai_format["function"]["arguments"]
    assert openai_format["id"].startswith("call_")


def test_parse_result_factories_set_success_payloads_and_default_warnings():
    tool_call = ToolCall(tool_name="test", arguments={})

    ok_result = ParseResult.ok([tool_call], warnings=None)
    err_result = ParseResult.error(["bad input"], warnings=None)

    assert ok_result.success is True
    assert ok_result.tool_calls == [tool_call]
    assert ok_result.errors == []
    assert ok_result.warnings == []

    assert err_result.success is False
    assert err_result.tool_calls == []
    assert err_result.errors == ["bad input"]
    assert err_result.warnings == []


def test_parse_result_is_frozen():
    result = ParseResult.ok([ToolCall(tool_name="test", arguments={})])

    with pytest.raises(FrozenInstanceError):
        result.success = False
