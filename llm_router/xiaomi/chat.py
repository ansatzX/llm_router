"""Xiaomi MiMo Chat Completions compatibility helpers.

Xiaomi MiMo exposes an OpenAI-compatible Chat Completions API, but its
documented request surface and thinking/tool replay behavior are provider
specific. The adapter owns Xiaomi-specific payload filtering, preserves
``developer`` messages, and shares only provider-neutral Responses/Chat
conversion helpers.
"""

from __future__ import annotations

import json
from typing import Any

from llm_router.chat_adapter_base import ChatCompletionAdapterBase
from llm_router.debug_log import log_debug


class XiaomiChatAdapter(ChatCompletionAdapterBase):
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
        converted: dict[str, Any] = {
            "type": "web_search",
            "max_keyword": 3,
            "force_search": True,
            "limit": 1,
        }
        for key in ("max_keyword", "limit"):
            if key in tool:
                converted[key] = tool[key]
        if "force_search" in tool:
            converted["force_search"] = tool["force_search"]
        elif "forced_search" in tool:
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
        sanitized = [self._sanitize_chat_tool(tool) for tool in converted]
        self._log_tool_diagnostics(tools, converted, sanitized)
        return sanitized

    def _sanitize_chat_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Constrain Xiaomi function schemas without changing hosted tools."""
        if tool.get("type") != "function":
            return tool
        function = tool.get("function")
        if not isinstance(function, dict):
            return tool
        sanitized = dict(tool)
        sanitized_function = dict(function)
        parameters = sanitized_function.get("parameters")
        if isinstance(parameters, dict):
            sanitized_function["parameters"] = self._sanitize_json_schema(parameters)[0]
        sanitized["function"] = sanitized_function
        return sanitized

    def _sanitize_json_schema(self, schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Return a Xiaomi-friendly schema and whether it was nullable."""
        any_of = schema.get("anyOf")
        if isinstance(any_of, list):
            non_null = [
                option for option in any_of
                if not (
                    isinstance(option, dict)
                    and option.get("type") == "null"
                )
            ]
            nullable = len(non_null) != len(any_of)
            selected = self._select_xiaomi_any_of_schema(non_null)
            if selected is not None:
                sanitized, child_nullable = self._sanitize_json_schema(selected)
                merged = {
                    key: value for key, value in schema.items()
                    if key not in {"anyOf", "type"}
                }
                merged.update({
                    key: value for key, value in sanitized.items()
                    if key not in merged
                })
                return merged, nullable or child_nullable

        sanitized = dict(schema)
        nullable = False
        schema_type = sanitized.get("type")
        if isinstance(schema_type, list) and "null" in schema_type:
            non_null_types = [item for item in schema_type if item != "null"]
            if len(non_null_types) == 1:
                sanitized["type"] = non_null_types[0]
                nullable = True
                schema_type = sanitized["type"]

        properties = sanitized.get("properties")
        nullable_properties: set[str] = set()
        if isinstance(properties, dict):
            sanitized_properties = {}
            for name, child_schema in properties.items():
                if isinstance(child_schema, dict):
                    sanitized_child, child_nullable = self._sanitize_json_schema(
                        child_schema,
                    )
                    sanitized_properties[name] = sanitized_child
                    if child_nullable:
                        nullable_properties.add(name)
                else:
                    sanitized_properties[name] = child_schema
            sanitized["properties"] = sanitized_properties

        required = sanitized.get("required")
        if isinstance(required, list) and nullable_properties:
            sanitized["required"] = [
                name for name in required
                if name not in nullable_properties
            ]

        items = sanitized.get("items")
        if isinstance(items, dict):
            sanitized["items"] = self._sanitize_json_schema(items)[0]

        if sanitized.get("type") == "object":
            sanitized["additionalProperties"] = False

        return sanitized, nullable

    def _select_xiaomi_any_of_schema(
        self,
        options: list[Any],
    ) -> dict[str, Any] | None:
        """Collapse JSON Schema unions to one Xiaomi-safe branch.

        Xiaomi's Chat API rejects complex Codex/MCP tool schemas with ``anyOf``.
        Prefer a string branch for string-or-array tool parameters because the
        model can still express a single value and avoid invalid upstream JSON.
        """
        dict_options = [option for option in options if isinstance(option, dict)]
        if not dict_options:
            return None
        for preferred_type in ("string", "integer", "number", "boolean", "object", "array"):
            for option in dict_options:
                if option.get("type") == preferred_type:
                    return option
        return dict_options[0]

    def _log_tool_diagnostics(
        self,
        input_tools: list[dict[str, Any]],
        converted_tools: list[dict[str, Any]],
        sanitized_tools: list[dict[str, Any]],
    ) -> None:
        function_names = [
            tool.get("function", {}).get("name", "")
            for tool in sanitized_tools
            if tool.get("type") == "function"
        ]
        log_debug("XIAOMI_CHAT_TOOL_DIAGNOSTICS", {
            "input_tool_count": len(input_tools),
            "forwarded_tool_count": len(sanitized_tools),
            "function_count": sum(
                1 for tool in sanitized_tools
                if tool.get("type") == "function"
            ),
            "web_search_count": sum(
                1 for tool in sanitized_tools
                if tool.get("type") == "web_search"
            ),
            "max_tool_name_len": max(
                (len(name) for name in function_names),
                default=0,
            ),
            "schema_bytes_before": self._tool_schema_bytes(converted_tools),
            "schema_bytes_after": self._tool_schema_bytes(sanitized_tools),
            "schemas_with_any_of_before": self._count_function_schemas_with(
                converted_tools,
                "anyOf",
            ),
            "schemas_with_any_of_after": self._count_function_schemas_with(
                sanitized_tools,
                "anyOf",
            ),
            "schemas_with_null_type_before": self._count_function_schemas_with(
                converted_tools,
                '"null"',
            ),
            "schemas_with_null_type_after": self._count_function_schemas_with(
                sanitized_tools,
                '"null"',
            ),
            "object_schemas_missing_additional_properties_before": (
                self._count_object_schemas_missing_additional_properties(
                    converted_tools,
                )
            ),
            "object_schemas_missing_additional_properties_after": (
                self._count_object_schemas_missing_additional_properties(
                    sanitized_tools,
                )
            ),
            "object_schemas_with_additional_properties_true_before": (
                self._count_object_schemas_with_additional_properties_true(
                    converted_tools,
                )
            ),
            "object_schemas_with_additional_properties_true_after": (
                self._count_object_schemas_with_additional_properties_true(
                    sanitized_tools,
                )
            ),
        })

    def _tool_schema_bytes(self, tools: list[dict[str, Any]]) -> int:
        return sum(
            len(json.dumps(
                tool.get("function", {}).get("parameters", {}),
                ensure_ascii=False,
            ))
            for tool in tools
            if tool.get("type") == "function"
        )

    def _count_function_schemas_with(
        self,
        tools: list[dict[str, Any]],
        marker: str,
    ) -> int:
        return sum(
            1 for tool in tools
            if tool.get("type") == "function"
            and marker in json.dumps(
                tool.get("function", {}).get("parameters", {}),
                ensure_ascii=False,
            )
        )

    def _count_object_schemas_missing_additional_properties(
        self,
        tools: list[dict[str, Any]],
    ) -> int:
        count = 0
        for tool in tools:
            if tool.get("type") != "function":
                continue
            parameters = tool.get("function", {}).get("parameters", {})
            count += self._count_object_schema_nodes_missing_additional_properties(
                parameters,
            )
        return count

    def _count_object_schema_nodes_missing_additional_properties(
        self,
        schema: Any,
    ) -> int:
        if isinstance(schema, list):
            return sum(
                self._count_object_schema_nodes_missing_additional_properties(item)
                for item in schema
            )
        if not isinstance(schema, dict):
            return 0
        count = 0
        if (
            schema.get("type") == "object"
            and "additionalProperties" not in schema
        ):
            count += 1
        for value in schema.values():
            count += self._count_object_schema_nodes_missing_additional_properties(
                value,
            )
        return count

    def _count_object_schemas_with_additional_properties_true(
        self,
        tools: list[dict[str, Any]],
    ) -> int:
        count = 0
        for tool in tools:
            if tool.get("type") != "function":
                continue
            parameters = tool.get("function", {}).get("parameters", {})
            count += self._count_object_schema_nodes_with_additional_properties_true(
                parameters,
            )
        return count

    def _count_object_schema_nodes_with_additional_properties_true(
        self,
        schema: Any,
    ) -> int:
        if isinstance(schema, list):
            return sum(
                self._count_object_schema_nodes_with_additional_properties_true(
                    item,
                )
                for item in schema
            )
        if not isinstance(schema, dict):
            return 0
        count = 0
        if (
            schema.get("type") == "object"
            and schema.get("additionalProperties") is True
        ):
            count += 1
        for value in schema.values():
            count += self._count_object_schema_nodes_with_additional_properties_true(
                value,
            )
        return count

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
