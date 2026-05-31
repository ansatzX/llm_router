"""DeepSeek provider-specific Chat Completions adapter."""

from llm_router.deepseek.anthropic_web_search import DeepSeekAnthropicWebSearchBridge
from llm_router.deepseek.chat import DeepSeekChatAdapter

__all__ = ["DeepSeekAnthropicWebSearchBridge", "DeepSeekChatAdapter"]
