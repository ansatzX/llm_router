"""Tests for MiroThinker MCP-first adapter behavior."""

from llm_router.mirothinker import MiroThinkerMCPAdapter


def _tool():
    return {
        "type": "function",
        "name": "exec_command",
        "description": "Runs a command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    }


def test_mirothinker_injects_mcp_prompt_and_forces_non_streaming():
    adapter = MiroThinkerMCPAdapter()
    payload = {
        "model": "mirothinker-test",
        "messages": [{"role": "user", "content": "list files"}],
        "stream": True,
    }

    adapter.prepare_payload(payload, [_tool()], server_name="tools")

    assert payload["stream"] is False
    assert payload["messages"][0]["role"] == "system"
    assert "<use_mcp_tool>" in payload["messages"][0]["content"]
    assert "exec_command" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "list files"}


def test_mirothinker_parses_reasoning_mcp_tool_call_to_openai_format():
    adapter = MiroThinkerMCPAdapter()
    result = adapter.parse_message(
        content="",
        reasoning_content="""
<use_mcp_tool>
  <server_name>tools</server_name>
  <tool_name>exec_command</tool_name>
  <arguments>{"cmd":"ls"}</arguments>
</use_mcp_tool>
""",
        tools=[_tool()],
    )

    assert result.success is True
    assert adapter.to_openai_tool_calls(result)[0]["function"] == {
        "name": "exec_command",
        "arguments": '{"cmd": "ls"}',
    }


def test_mirothinker_converts_parsed_calls_to_responses_output_items():
    adapter = MiroThinkerMCPAdapter()
    result = adapter.parse_message(
        content="""
<use_mcp_tool>
  <server_name>tools</server_name>
  <tool_name>exec_command</tool_name>
  <arguments>{"cmd":"ls"}</arguments>
</use_mcp_tool>
""",
        reasoning_content="",
        tools=[_tool()],
    )

    items = adapter.to_responses_output_items(result)

    assert items[0]["type"] == "function_call"
    assert items[0]["id"].startswith("call_")
    assert items[0]["call_id"] == items[0]["id"]
    assert items[0]["name"] == "exec_command"
    assert items[0]["arguments"] == '{"cmd": "ls"}'


def test_mirothinker_builds_responses_output_with_matching_tool_call_ids():
    adapter = MiroThinkerMCPAdapter()
    result = adapter.parse_message(
        content="""
<use_mcp_tool>
  <server_name>tools</server_name>
  <tool_name>exec_command</tool_name>
  <arguments>{"cmd":"ls"}</arguments>
</use_mcp_tool>
""",
        reasoning_content="",
        tools=[_tool()],
    )

    output_items, tool_calls = adapter.to_responses_tool_outputs(result)

    assert output_items[0]["id"] == tool_calls[0]["id"]
    assert output_items[0]["call_id"] == tool_calls[0]["id"]


def test_mirothinker_appends_retry_feedback_for_incomplete_mcp_xml():
    adapter = MiroThinkerMCPAdapter()
    payload = {"messages": [{"role": "user", "content": "run ls"}]}
    result = adapter.parse_message(
        content="<use_mcp_tool><tool_name>exec_command</tool_name>",
        reasoning_content="",
        tools=[_tool()],
    )

    assert adapter.should_retry(
        result,
        response_text="<use_mcp_tool><tool_name>exec_command</tool_name>",
        retry_count=0,
        max_retries=3,
    )

    adapter.append_retry_feedback(
        payload,
        response_text="<use_mcp_tool><tool_name>exec_command</tool_name>",
        errors=result.errors,
    )

    assert payload["messages"][-2]["role"] == "assistant"
    assert payload["messages"][-1]["role"] == "user"
    assert "[RETRY INSTRUCTION]" in payload["messages"][-1]["content"]
