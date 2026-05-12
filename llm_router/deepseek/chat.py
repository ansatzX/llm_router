"""DeepSeek Chat Completions compatibility helpers."""

from __future__ import annotations

from typing import Any

from llm_router.chat_adapter_base import ChatCompletionAdapterBase
from llm_router.debug_log import log_debug


class DeepSeekChatAdapter(ChatCompletionAdapterBase):
    """Adapter for DeepSeek's official OpenAI-compatible Chat API."""

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
    COMPAT_GATEWAY_REQUEST_PARAMS = {
        "prompt_cache_key",
        "prompt_cache_retention",
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
