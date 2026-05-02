"""Helpers for OpenAI-compatible Chat Completions backends."""

from __future__ import annotations

from typing import Any

from llm_router.debug_log import log_debug
from llm_router.deepseek.chat import DeepSeekChatAdapter


class OpenAIChatAdapter(DeepSeekChatAdapter):
    """Adapter for generic OpenAI-compatible Chat Completions APIs."""

    CHAT_REQUEST_PARAMS = {
        "model",
        "messages",
        "stream",
        "temperature",
        "top_p",
        "n",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "reasoning_effort",
        "service_tier",
        "metadata",
        "modalities",
        "prediction",
        "audio",
        "safety_identifier",
        "prompt_cache_retention",
        "verbosity",
        "web_search_options",
        "store",
    }

    DROPPED_REQUEST_PARAMS = {
        "client_metadata",
        "prompt_cache_key",
        "reasoning",
    }

    def filter_request_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Drop Responses-only fields and map text output controls to chat shape."""
        filtered = {
            key: value
            for key, value in payload.items()
            if key in self.CHAT_REQUEST_PARAMS
        }
        text_spec = payload.get("text")
        reasoning_effort = self._reasoning_effort_from_payload(payload)

        for key in self.DROPPED_REQUEST_PARAMS:
            filtered.pop(key, None)
        if reasoning_effort:
            filtered["reasoning_effort"] = reasoning_effort

        response_format = self._text_to_response_format(text_spec)
        if response_format is not None:
            filtered["response_format"] = response_format

        dropped = sorted(
            key for key in payload
            if key not in filtered and key != "text"
        )
        if text_spec is not None:
            dropped.append("text")
        if dropped:
            log_debug("OPENAI_CHAT_PAYLOAD_FILTER", {
                "dropped_keys": sorted(set(dropped)),
                "forwarded_keys": sorted(filtered),
                "mapped_response_format": response_format,
            })
        return filtered

    def _text_to_response_format(self, text_spec: Any) -> dict[str, Any] | None:
        """Convert Responses API text.format config to Chat Completions response_format."""
        if not isinstance(text_spec, dict):
            return None
        fmt = text_spec.get("format")
        if not isinstance(fmt, dict):
            return None

        fmt_type = fmt.get("type")
        if fmt_type == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": fmt,
            }
        if fmt_type in {"json_object", "text"}:
            return {"type": fmt_type}
        return None
