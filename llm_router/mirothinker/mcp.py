"""MiroThinker MCP-first Chat helpers.

MiroThinker does not use native Chat tool calls. It receives Codex tools as an
MCP XML instruction prompt, emits XML tool calls in content or reasoning text,
and this adapter converts those parsed calls back to OpenAI/Codex tool calls.
"""

from __future__ import annotations

from typing import Any

from llm_router.debug_log import log_debug
from llm_router.mcp_converter import generate_mcp_system_prompt
from llm_router.parser import ParseResult, parse_tool_calls


class MiroThinkerMCPAdapter:
    """Adapter for MCP-first MiroThinker requests."""

    def prepare_payload(
        self,
        payload: dict[str, Any],
        tools: list[dict[str, Any]],
        server_name: str,
    ) -> None:
        """Inject MCP XML instructions and force non-streaming processing."""
        if not tools:
            return

        tool_names = []
        for tool in tools:
            if tool.get("type") == "function":
                tool_names.append(tool.get("name") or tool.get("function", {}).get("name"))
            elif "name" in tool:
                tool_names.append(tool.get("name"))

        log_debug("MIROTHINKER_MCP_PARSE_CONTEXT", {
            "model": payload.get("model"),
            "available_tool_names": [name for name in tool_names if name],
            "tool_count": len(tools),
        })

        messages = list(payload.get("messages", []))
        prompt = generate_mcp_system_prompt(tools, server_name)
        self.inject_system_prompt(messages, prompt)
        payload["messages"] = messages
        payload["stream"] = False

    def parse_message(
        self,
        content: str,
        reasoning_content: str,
        tools: list[dict[str, Any]],
    ) -> ParseResult:
        """Parse MCP XML tool calls from assistant visible or reasoning text."""
        result = parse_tool_calls(
            content,
            reasoning_content,
            available_tools=tools,
        )
        log_debug("MIROTHINKER_MCP_PARSE_RESULT", {
            "success": result.success,
            "tool_calls": [
                {"name": tc.tool_name, "arguments": tc.arguments}
                for tc in result.tool_calls
            ],
            "errors": result.errors,
            "warnings": result.warnings,
            "response_text": content,
            "reasoning_text": reasoning_content,
        }, truncate=False)
        return result

    def should_retry(
        self,
        result: ParseResult,
        response_text: str,
        retry_count: int,
        max_retries: int,
    ) -> bool:
        """Return whether malformed MCP XML should trigger rollback retry."""
        return (
            not result.success
            and bool(result.errors)
            and self.has_incomplete_mcp_tags(response_text)
            and retry_count < max_retries - 1
        )

    def append_retry_feedback(
        self,
        payload: dict[str, Any],
        response_text: str,
        errors: list[str],
    ) -> None:
        """Append failed assistant output and retry instruction to the payload."""
        assistant_msg = {"role": "assistant", "content": response_text or ""}
        if assistant_msg["content"]:
            payload.setdefault("messages", []).append(assistant_msg)

        payload.setdefault("messages", []).append({
            "role": "user",
            "content": self.format_parse_errors(errors),
        })

    def to_openai_tool_calls(self, result: ParseResult) -> list[dict[str, Any]]:
        """Convert parsed MCP calls into OpenAI Chat tool call objects."""
        return [tool_call.to_openai_format() for tool_call in result.tool_calls]

    def to_responses_output_items(
        self,
        result: ParseResult,
    ) -> list[dict[str, Any]]:
        """Convert parsed MCP calls into Responses function_call items."""
        output_items, _ = self.to_responses_tool_outputs(result)
        return output_items

    def to_responses_tool_outputs(
        self,
        result: ParseResult,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build matching Responses output items and Chat-style tool calls."""
        items = []
        tool_calls = []
        for tool_call in result.tool_calls:
            openai_call = tool_call.to_openai_format()
            tool_calls.append(openai_call)
            function = openai_call["function"]
            call_id = openai_call["id"]
            items.append({
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": function["name"],
                "arguments": function["arguments"],
            })
        return items, tool_calls

    def inject_system_prompt(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
    ) -> list[dict[str, Any]]:
        """Inject system prompt at the beginning of messages."""
        if not prompt:
            return messages
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{prompt}\n\n{messages[0]['content']}"
        else:
            messages.insert(0, {"role": "system", "content": prompt})
        return messages

    def has_incomplete_mcp_tags(self, text: str) -> bool:
        """Check if text contains partial MCP XML markup."""
        if not text:
            return False
        patterns = [
            "<use_mcp_tool>", "</use_mcp_tool>",
            "<server_name>", "</server_name>",
            "<tool_name>", "</tool_name>",
            "<arguments>", "</arguments>",
        ]
        return any(pattern in text for pattern in patterns)

    def format_parse_errors(self, errors: list[str]) -> str:
        """Format MCP parse errors as retry feedback for the model."""
        if not errors:
            return "Unknown parsing error"
        formatted = "\n".join(f"  - {error}" for error in errors)
        return (
            "[RETRY INSTRUCTION]\n"
            f"The previous tool call had parsing errors:\n{formatted}\n\n"
            "Please regenerate the tool call with correct format. "
            "Do not apologize - just output the corrected tool call."
        )
