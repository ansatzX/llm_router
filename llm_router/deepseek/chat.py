"""DeepSeek Chat Completions compatibility helpers.

DeepSeek's OpenAI-compatible Chat API accepts only ``tools[].type == "function"``,
while Codex can send Responses API tools such as ``custom``.
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
    HOSTED_TOOL_TYPES = {
        "web_search",
        "file_search",
        "image_generation",
        "computer_use_preview",
    }

    COMPAT_GATEWAY_REQUEST_PARAMS = {
        "prompt_cache_key",
        "prompt_cache_retention",
    }

    def __init__(self, forward_compat_prompt_cache: bool = False) -> None:
        self.forward_compat_prompt_cache = forward_compat_prompt_cache
        self.reasoning_by_call_id: dict[str, str] = {}

    def reset(self) -> None:
        self.reasoning_by_call_id.clear()

    def load_provider_state(self, provider_state: dict[str, Any] | None) -> None:
        """Load persisted DeepSeek-private state for one Responses session."""
        self.reset()
        if not isinstance(provider_state, dict):
            return
        reasoning_by_call_id = provider_state.get("reasoning_by_call_id", {})
        if isinstance(reasoning_by_call_id, dict):
            self.reasoning_by_call_id.update({
                str(key): str(value)
                for key, value in reasoning_by_call_id.items()
                if value
            })

    def dump_provider_state(self) -> dict[str, dict[str, str]]:
        """Return DeepSeek-private state suitable for session persistence."""
        return {
            "reasoning_by_call_id": dict(self.reasoning_by_call_id),
        }

    def filter_request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only request fields DeepSeek's Chat Completion API accepts."""
        reasoning_effort = self._reasoning_effort_from_payload(payload)
        allowed_params = set(self.CHAT_REQUEST_PARAMS)
        if self.forward_compat_prompt_cache:
            allowed_params.update(self.COMPAT_GATEWAY_REQUEST_PARAMS)
        filtered = {
            key: value
            for key, value in payload.items()
            if key in allowed_params
        }
        if reasoning_effort and "reasoning_effort" not in filtered:
            filtered["reasoning_effort"] = reasoning_effort
        dropped = sorted(set(payload) - set(filtered))
        if dropped:
            log_debug("DEEPSEEK_CHAT_PAYLOAD_FILTER", {
                "dropped_keys": dropped,
                "forwarded_keys": sorted(filtered),
            })
        return filtered

    def _reasoning_effort_from_payload(
        self,
        payload: dict[str, Any],
    ) -> str | None:
        """Extract chat-compatible reasoning effort from a Responses payload."""
        reasoning_effort = payload.get("reasoning_effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            return reasoning_effort

        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if isinstance(effort, str) and effort:
                return effort
        return None

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
            namespace = msg.get("namespace")
            if (
                msg_type == "function_call"
                and isinstance(namespace, str)
                and namespace
            ):
                name = self.namespaced_chat_tool_name(namespace, str(name))
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

            flush_pending_tool_calls()
            if isinstance(converted, list):
                result.extend(converted)
            else:
                result.append(converted)
        flush_pending_tool_calls()
        log_debug("DEEPSEEK_RESPONSES_TO_CHAT_CONVERSION", {
            "input_count": len(messages),
            "input_types": input_types,
            "output_count": len(result),
            "empty_output_messages": sum(1 for m in result if not m.get("content")),
            "output_roles": [m.get("role") for m in result],
        })
        return result

    def responses_tools_to_chat(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert all Codex Responses tools to DeepSeek Chat function tools."""
        converted = []
        for tool in tools:
            if tool.get("type") in self.HOSTED_TOOL_TYPES:
                continue
            if tool.get("type") == "function":
                converted.append(self.response_tool_to_chat(tool))
                continue
            if tool.get("type") == "namespace":
                converted.extend(self.namespace_tool_to_chat(tool))
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

    def namespace_tool_to_chat(self, tool: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand a Responses namespace into DeepSeek Chat function tools."""
        namespace = tool.get("name")
        if not isinstance(namespace, str) or not namespace:
            return []
        converted = []
        for child in tool.get("tools", []):
            if not isinstance(child, dict) or child.get("type") != "function":
                continue
            child_name = child.get("name")
            if not isinstance(child_name, str) or not child_name:
                continue
            converted.append({
                "type": "function",
                "function": {
                    "name": self.namespaced_chat_tool_name(namespace, child_name),
                    "description": child.get("description", ""),
                    "parameters": child.get("parameters", {}),
                },
            })
        return converted

    def namespaced_chat_tool_name(self, namespace: str, name: str) -> str:
        """Return Codex's flat model-visible name for a namespaced tool."""
        if namespace.endswith("_") or name.startswith("_"):
            return f"{namespace}{name}"
        return f"{namespace}_{name}"

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

    def tool_type_map(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Map Chat function names back to original Responses tool types."""
        mapping = {}
        for tool in tools:
            name = tool.get("name")
            if not name and tool.get("type") == "function":
                name = tool.get("function", {}).get("name")
            if name:
                mapping[name] = tool.get("type", "function")
            if tool.get("type") == "namespace" and isinstance(name, str):
                for child in tool.get("tools", []):
                    if not isinstance(child, dict) or child.get("type") != "function":
                        continue
                    child_name = child.get("name")
                    if not isinstance(child_name, str) or not child_name:
                        continue
                    mapping[self.namespaced_chat_tool_name(name, child_name)] = {
                        "type": "namespace_function",
                        "namespace": name,
                        "name": child_name,
                    }
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
        if reasoning_content:
            output_items.append({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_content}],
                "content": [{"type": "reasoning_text", "text": reasoning_content}],
            })

        if response_text:
            item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": response_text}],
            }
            if reasoning_content:
                item["reasoning_content"] = reasoning_content
            output_items.append(item)

        if native_tool_calls:
            output_items = [
                item for item in output_items
                if item.get("type") == "reasoning"
            ]
            for tool_call in native_tool_calls:
                call_id = tool_call.get("id", "")
                self.record_tool_reasoning(call_id, reasoning_content)
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                arguments = function.get("arguments", "")

                tool_mapping = tool_type_map.get(tool_name)
                if tool_mapping == "custom":
                    item = {
                        "type": "custom_tool_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": tool_name,
                        "input": self.chat_tool_arguments_to_custom_input(arguments),
                    }
                elif (
                    isinstance(tool_mapping, dict)
                    and tool_mapping.get("type") == "namespace_function"
                ):
                    item = {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "namespace": tool_mapping.get("namespace", ""),
                        "name": tool_mapping.get("name", tool_name),
                        "arguments": arguments,
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

    def record_tool_reasoning(
        self,
        call_id: str,
        reasoning_content: str | None,
    ) -> None:
        if call_id and reasoning_content:
            self.reasoning_by_call_id[call_id] = reasoning_content
