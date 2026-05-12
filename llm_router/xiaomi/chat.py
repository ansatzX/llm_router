"""Xiaomi MiMo Chat Completions compatibility helpers.

Xiaomi MiMo exposes an OpenAI-compatible Chat Completions API, but its
documented request surface and thinking/tool replay behavior are provider
specific. The adapter keeps Xiaomi-specific payload filtering and preserves
``developer`` messages, while reusing the Responses tool/reasoning conversion
needed by Codex.
"""

from __future__ import annotations

from typing import Any

from llm_router.debug_log import log_debug
from llm_router.deepseek import DeepSeekChatAdapter


class XiaomiChatAdapter(DeepSeekChatAdapter):
    """Adapter for Xiaomi MiMo's official OpenAI-compatible Chat API."""

    CHAT_REQUEST_PARAMS = {
        "model",
        "messages",
        "stream",
        "temperature",
        "top_p",
        "max_completion_tokens",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "stop",
        "tools",
        "tool_choice",
        "thinking",
    }

    def filter_request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep only request fields Xiaomi's official Chat API documents."""
        filtered = {
            key: value
            for key, value in payload.items()
            if key in self.CHAT_REQUEST_PARAMS
        }
        if "thinking" not in filtered:
            thinking = self._thinking_from_payload(payload)
            if thinking is not None:
                filtered["thinking"] = thinking
        if filtered.get("tool_choice") != "auto":
            filtered.pop("tool_choice", None)

        dropped = sorted(set(payload) - set(filtered))
        if dropped:
            log_debug("XIAOMI_CHAT_PAYLOAD_FILTER", {
                "dropped_keys": dropped,
                "forwarded_keys": sorted(filtered),
            })
        return filtered

    def _thinking_from_payload(self, payload: dict[str, Any]) -> dict[str, str] | None:
        """Map Codex reasoning controls to Xiaomi's documented thinking knob."""
        effort = self._reasoning_effort_from_payload(payload)
        if not effort:
            return None
        thinking_type = "disabled" if effort in {"none", "minimal"} else "enabled"
        return {"type": thinking_type}

    def content_item_to_chat_part(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Convert one Responses content item to Xiaomi Chat content part."""
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type in ("input_text", "output_text", "text"):
            text = item.get("text")
            if isinstance(text, str):
                return {"type": "text", "text": text}
            return None
        if item_type in ("input_image", "image_url"):
            image_url = item.get("image_url") or item.get("url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url:
                return {"type": "image_url", "image_url": {"url": image_url}}
        return None

    def content_items_to_chat_content(self, content: list[Any]) -> str | list[dict[str, Any]]:
        """Preserve Xiaomi-supported image parts while keeping text-only compact."""
        parts = [
            part for item in content
            if isinstance(item, dict)
            for part in [self.content_item_to_chat_part(item)]
            if part is not None
        ]
        has_image = any(part.get("type") == "image_url" for part in parts)
        if has_image:
            return parts
        return "\n".join(
            str(part.get("text", ""))
            for part in parts
            if part.get("type") == "text" and part.get("text")
        )

    def output_to_chat_content(self, output: Any) -> str | list[dict[str, Any]]:
        """Convert Responses tool output to Xiaomi Chat content."""
        if isinstance(output, list):
            content = self.content_items_to_chat_content(output)
            if content != "":
                return content
        return self.output_to_text(output)

    def xiaomi_web_search_tool_to_chat(
        self,
        tool: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Convert Codex/OpenAI web_search options to Xiaomi's documented shape."""
        if tool.get("external_web_access") is False:
            return None
        converted: dict[str, Any] = {"type": "web_search"}
        for key in ("max_keyword", "force_search", "limit"):
            if key in tool:
                converted[key] = tool[key]
        if "force_search" not in converted and "forced_search" in tool:
            converted["force_search"] = tool["forced_search"]

        user_location = tool.get("user_location")
        if isinstance(user_location, dict):
            converted_location = {
                key: user_location[key]
                for key in ("type", "country", "region", "city")
                if key in user_location
            }
            if converted_location:
                converted["user_location"] = converted_location
        return converted

    def responses_tools_to_chat(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Codex Responses tools to Xiaomi Chat tools."""
        converted = []
        for tool in tools:
            if tool.get("type") == "web_search":
                web_search_tool = self.xiaomi_web_search_tool_to_chat(tool)
                if web_search_tool:
                    converted.append(web_search_tool)
                continue
            converted.extend(super().responses_tools_to_chat([tool]))
        return converted

    def response_item_to_chat(
        self,
        msg: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert one Responses item to Xiaomi Chat message form.

        Xiaomi documents ``developer`` as a supported role, so unlike DeepSeek
        this adapter must not down-convert it to ``system``.
        """
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
            arguments = self.response_item_tool_arguments_to_chat(msg)
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
                "content": self.output_to_chat_content(msg.get("output")),
            }

        role = msg.get("role", "user")
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

            chat_content = self.content_items_to_chat_content(content)
            return self._message_with_reasoning(role, chat_content, msg)

        return {"role": role, "content": str(content) if content else ""}

    def chat_response_to_output_items(
        self,
        message: dict[str, Any],
        tool_type_map: dict[str, str],
    ) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]]]:
        """Convert Xiaomi Chat response and preserve web-search annotations."""
        output_items, output_text, native_tool_calls = super().chat_response_to_output_items(
            message,
            tool_type_map,
        )
        annotations = message.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            return output_items, output_text, native_tool_calls

        output_items.insert(0, {
            "type": "web_search_call",
            "status": "completed",
        })
        for item in output_items:
            if item.get("type") != "message":
                continue
            for content_item in item.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "output_text"
                ):
                    content_item["annotations"] = annotations
        return output_items, output_text, native_tool_calls
