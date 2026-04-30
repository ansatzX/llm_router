"""DeepSeek Chat Completions compatibility helpers.

DeepSeek's OpenAI-compatible Chat API accepts only ``tools[].type == "function"``
but Codex sends Responses API tools such as ``custom`` and ``web_search``.
DeepSeek thinking mode also requires assistant ``reasoning_content`` to be
included in later multi-round requests.
"""

from __future__ import annotations

import json
from typing import Any

from llm_router.debug_log import log_debug


class DeepSeekChatAdapter:
    """Adapter for DeepSeek's OpenAI-compatible Chat API."""

    CHAT_REQUEST_PARAMS = {
        "model",
        "messages",
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "thinking",
        "reasoning_effort",
    }

    def __init__(self) -> None:
        self.reasoning_by_call_id: dict[str, str] = {}
        self.reasoning_by_message_content: dict[str, str] = {}

    def reset(self) -> None:
        self.reasoning_by_call_id.clear()
        self.reasoning_by_message_content.clear()

    def filter_request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only request fields DeepSeek's Chat Completion API accepts."""
        filtered = {
            key: value
            for key, value in payload.items()
            if key in self.CHAT_REQUEST_PARAMS
        }
        dropped = sorted(set(payload) - set(filtered))
        if dropped:
            log_debug("DEEPSEEK_CHAT_PAYLOAD_FILTER", {
                "dropped_keys": dropped,
                "forwarded_keys": sorted(filtered),
            })
        return filtered

    def content_item_to_text(self, item: dict[str, Any]) -> str:
        """Extract text from a Responses API content item."""
        if not isinstance(item, dict):
            return ""
        item_type = item.get("type")
        if item_type in ("input_text", "output_text", "text"):
            return item.get("text", "")
        return ""

    def output_to_text(self, output: Any) -> str:
        """Convert Responses tool output payloads into Chat-visible text."""
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            parts = []
            for item in output:
                if isinstance(item, dict):
                    text = self.content_item_to_text(item)
                    if not text and "text" in item:
                        text = str(item["text"])
                    if text:
                        parts.append(text)
                elif item is not None:
                    parts.append(str(item))
            if parts:
                return "\n".join(parts)
        try:
            return json.dumps(output, ensure_ascii=False)
        except TypeError:
            return str(output)

    def response_item_to_chat(
        self,
        msg: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert one Responses item to Chat message form."""
        if not isinstance(msg, dict):
            return {"role": "user", "content": ""}

        msg_type = msg.get("type")
        if msg_type in ("function_call", "custom_tool_call"):
            name = msg.get("name", "unknown_tool")
            call_id = msg.get("call_id") or msg.get("id") or "unknown_call"
            arguments = msg.get("arguments", msg.get("input", ""))
            converted = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    },
                ],
            }
            reasoning_content = (
                msg.get("reasoning_content")
                or self.reasoning_by_call_id.get(call_id)
            )
            if reasoning_content:
                converted["reasoning_content"] = reasoning_content
            return converted

        if msg_type in ("function_call_output", "custom_tool_call_output"):
            call_id = msg.get("call_id") or "unknown_call"
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": self.output_to_text(msg.get("output")),
            }

        role = msg.get("role", "user")
        if role == "developer":
            role = "system"
        content = msg.get("content")

        if isinstance(content, str):
            return self._message_with_reasoning(role, content, msg)

        if isinstance(content, list):
            nested = [
                c for c in content
                if isinstance(c, dict) and c.get("type") == "message"
            ]
            if nested:
                result = []
                for n in nested:
                    converted = self.response_item_to_chat(n)
                    if isinstance(converted, list):
                        result.extend(converted)
                    else:
                        result.append(converted)
                return result

            text_content = "\n".join(
                t for t in (self.content_item_to_text(c) for c in content) if t
            )
            return self._message_with_reasoning(role, text_content, msg)

        return {"role": role, "content": str(content) if content else ""}

    def _message_with_reasoning(
        self,
        role: str,
        content: str,
        source: dict[str, Any],
    ) -> dict[str, Any]:
        converted = {"role": role, "content": content}
        reasoning_content = source.get("reasoning_content")
        if (
            not reasoning_content
            and role == "assistant"
            and content in self.reasoning_by_message_content
        ):
            reasoning_content = self.reasoning_by_message_content[content]
        if reasoning_content:
            converted["reasoning_content"] = reasoning_content
        return converted

    def flatten_response_items(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Flatten Responses items into DeepSeek Chat messages.

        Adjacent Responses ``function_call`` items represent one parallel Chat
        assistant turn and must be grouped into one message before the tool
        output messages.
        """
        result = []
        input_types: dict[str, int] = {}
        pending_tool_call_message: dict[str, Any] | None = None
        pending_tool_call_emitted = False
        pending_interleaved_messages: list[dict[str, Any]] = []
        pending_legacy_tool_calls: list[dict[str, Any]] = []
        pending_legacy_outputs: list[dict[str, Any]] = []

        def flush_pending_tool_calls() -> None:
            nonlocal pending_tool_call_message, pending_tool_call_emitted
            nonlocal pending_interleaved_messages
            if pending_tool_call_message is not None:
                if not pending_tool_call_emitted:
                    result.append(pending_tool_call_message)
                if pending_interleaved_messages:
                    result.extend(pending_interleaved_messages)
                pending_tool_call_message = None
                pending_tool_call_emitted = False
                pending_interleaved_messages = []

        def pending_tool_call_ids() -> set[str]:
            if pending_tool_call_message is None:
                return set()
            return {
                tool_call.get("id", "")
                for tool_call in pending_tool_call_message.get("tool_calls", [])
            }

        def pending_tool_output_ids() -> set[str]:
            if pending_tool_call_message is None:
                return set()
            pending_ids = pending_tool_call_ids()
            output_ids = set()
            for item in result:
                if item.get("role") == "tool":
                    tool_call_id = item.get("tool_call_id", "")
                    if tool_call_id in pending_ids:
                        output_ids.add(tool_call_id)
            return output_ids

        def flush_legacy_tool_calls_without_outputs() -> None:
            nonlocal pending_legacy_tool_calls
            for tool_call in pending_legacy_tool_calls:
                result.append({
                    "role": "user",
                    "content": self._legacy_tool_call_to_text(tool_call),
                })
            pending_legacy_tool_calls = []

        def flush_legacy_tool_calls_with_outputs(
            outputs: list[dict[str, Any]],
        ) -> None:
            nonlocal pending_legacy_tool_calls, pending_legacy_outputs
            output_by_call_id = {
                output.get("tool_call_id") or "unknown_call": output
                for output in outputs
            }
            for tool_call in pending_legacy_tool_calls:
                call_id = tool_call["tool_calls"][0]["id"]
                output = output_by_call_id.get(call_id)
                result.append({
                    "role": "user",
                    "content": self._legacy_tool_call_to_text(tool_call, output),
                })
            pending_legacy_tool_calls = []
            pending_legacy_outputs = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            msg_type = msg.get("type", "<missing>")
            input_types[msg_type] = input_types.get(msg_type, 0) + 1
            converted = self.response_item_to_chat(msg)
            if (
                isinstance(converted, dict)
                and converted.get("role") == "assistant"
                and converted.get("tool_calls")
            ):
                if pending_legacy_outputs:
                    flush_legacy_tool_calls_with_outputs(pending_legacy_outputs)
                if "reasoning_content" not in converted:
                    flush_pending_tool_calls()
                    pending_legacy_tool_calls.append(converted)
                    continue
                flush_legacy_tool_calls_without_outputs()
                if pending_tool_call_message is None:
                    pending_tool_call_message = converted
                else:
                    pending_tool_call_message.setdefault("tool_calls", []).extend(
                        converted.get("tool_calls", []),
                    )
                    if (
                        "reasoning_content" not in pending_tool_call_message
                        and converted.get("reasoning_content")
                    ):
                        pending_tool_call_message["reasoning_content"] = (
                            converted["reasoning_content"]
                        )
                continue

            if (
                pending_legacy_tool_calls
                and isinstance(converted, dict)
                and converted.get("role") == "tool"
            ):
                pending_legacy_outputs.append(converted)
                continue

            if (
                pending_tool_call_message is not None
                and isinstance(converted, dict)
                and converted.get("role") == "tool"
                and converted.get("tool_call_id") in pending_tool_call_ids()
            ):
                if not pending_tool_call_emitted:
                    result.append(pending_tool_call_message)
                    pending_tool_call_emitted = True
                result.append(converted)
                if pending_tool_output_ids() >= pending_tool_call_ids():
                    if pending_interleaved_messages:
                        result.extend(pending_interleaved_messages)
                        pending_interleaved_messages = []
                    pending_tool_call_message = None
                    pending_tool_call_emitted = False
                continue

            if pending_tool_call_message is not None:
                if isinstance(converted, list):
                    pending_interleaved_messages.extend(converted)
                else:
                    pending_interleaved_messages.append(converted)
                continue

            if pending_legacy_outputs:
                flush_legacy_tool_calls_with_outputs(pending_legacy_outputs)
            flush_pending_tool_calls()
            if isinstance(converted, list):
                result.extend(converted)
            else:
                result.append(converted)
        if pending_legacy_outputs:
            flush_legacy_tool_calls_with_outputs(pending_legacy_outputs)
        flush_legacy_tool_calls_without_outputs()
        flush_pending_tool_calls()
        log_debug("DEEPSEEK_RESPONSES_TO_CHAT_CONVERSION", {
            "input_count": len(messages),
            "input_types": input_types,
            "output_count": len(result),
            "empty_output_messages": sum(1 for m in result if not m.get("content")),
            "output_roles": [m.get("role") for m in result],
        })
        return result

    def _legacy_tool_call_to_text(
        self,
        tool_call_message: dict[str, Any],
        tool_output_message: dict[str, Any] | None = None,
    ) -> str:
        tool_call = tool_call_message["tool_calls"][0]
        function = tool_call.get("function", {})
        text = (
            f"[historical tool call omitted: {function.get('name', '')} "
            f"call_id={tool_call.get('id', '')}]\n"
            f"{function.get('arguments', '')}"
        )
        if tool_output_message is not None:
            text += f"\nTool output:\n{tool_output_message.get('content', '')}"
        return text

    def responses_tools_to_chat(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert all Codex Responses tools to DeepSeek Chat function tools."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                converted.append(self.response_tool_to_chat(tool))
                continue
            name = tool.get("name") or tool.get("type", "unknown_tool")
            converted.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Freeform input for the custom tool.",
                            },
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            })
        log_debug("DEEPSEEK_CHAT_TOOL_CONVERSION", {
            "input_tools": [
                {"type": t.get("type"), "name": t.get("name")}
                for t in tools
            ],
            "forwarded_tool_names": [
                t.get("function", {}).get("name") for t in converted
            ],
        })
        return converted

    def response_tool_to_chat(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert a Responses function tool to Chat function format."""
        if tool.get("type") == "function":
            if "function" in tool:
                return tool
            if "name" in tool:
                return {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
        return tool

    def tool_type_map(self, tools: list[dict[str, Any]]) -> dict[str, str]:
        """Map Chat function names back to original Responses tool types."""
        mapping = {}
        for tool in tools:
            name = tool.get("name")
            if not name and tool.get("type") == "function":
                name = tool.get("function", {}).get("name")
            if name:
                mapping[name] = tool.get("type", "function")
        return mapping

    def chat_tool_arguments_to_custom_input(self, arguments: Any) -> str:
        """Extract freeform custom input from wrapped Chat function arguments."""
        if not isinstance(arguments, str):
            return str(arguments or "")
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
        if isinstance(parsed, dict) and "input" in parsed:
            return str(parsed["input"])
        return arguments

    def chat_response_to_output_items(
        self,
        message: dict[str, Any],
        tool_type_map: dict[str, str],
    ) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]]]:
        """Convert a DeepSeek Chat assistant message into Responses output.

        DeepSeek returns native Chat ``tool_calls`` even for Codex Responses
        tools that had to be wrapped as Chat functions. Restore the original
        Responses item type before sending the result back to Codex.
        """
        response_text = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")
        native_tool_calls = message.get("tool_calls") or []

        output_items: list[dict[str, Any]] = []
        if response_text:
            item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response_text}],
            }
            if reasoning_content:
                item["reasoning_content"] = reasoning_content
                self.record_message_reasoning(response_text, reasoning_content)
            output_items.append(item)

        if native_tool_calls:
            output_items = []
            for tool_call in native_tool_calls:
                call_id = tool_call.get("id", "")
                self.record_tool_reasoning(call_id, reasoning_content)
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                arguments = function.get("arguments", "")

                if tool_type_map.get(tool_name) == "custom":
                    item = {
                        "type": "custom_tool_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": tool_name,
                        "input": self.chat_tool_arguments_to_custom_input(arguments),
                    }
                else:
                    item = {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                    }
                if reasoning_content:
                    item["reasoning_content"] = reasoning_content
                output_items.append(item)

        output_text = None if native_tool_calls else response_text
        return output_items, output_text, native_tool_calls

    def record_message_reasoning(
        self,
        content: str,
        reasoning_content: str | None,
    ) -> None:
        if content and reasoning_content:
            self.reasoning_by_message_content[content] = reasoning_content

    def record_tool_reasoning(
        self,
        call_id: str,
        reasoning_content: str | None,
    ) -> None:
        if call_id and reasoning_content:
            self.reasoning_by_call_id[call_id] = reasoning_content
