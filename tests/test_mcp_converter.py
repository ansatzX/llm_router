"""Tests for MCP prompt generation."""

from llm_router.mcp_converter import generate_mcp_system_prompt


def test_mcp_prompt_tells_model_not_to_repeat_bootstrap_or_invent_tools():
    """The MCP prompt should guide non-native tool models away from bad loops."""
    prompt = generate_mcp_system_prompt(
        [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Runs a command",
                "parameters": {"type": "object"},
            }
        ],
        server_name="tools",
    )

    assert "Do not repeat or quote bootstrapping instructions" in prompt
    assert "do not invent shorthand tools such as ls, cat, rg, or read" in prompt
    assert "Do not repeat the same tool call" in prompt
