"""Tests for TOML router configuration."""

from llm_router.config import RouterConfig


def test_deepseek_models_route_as_plain_chat():
    """DeepSeek should keep Codex function/tool protocol, not MCP-first XML."""
    cfg = RouterConfig.from_toml("router.toml")

    model_type, upstream = cfg.resolve("deepseek-chat")

    assert model_type == "chat"
    assert upstream.base_url == "https://api.deepseek.com"


def test_only_mirothinker_routes_through_mcp_parser():
    """MiroThinker is the only configured MCP-first provider."""
    cfg = RouterConfig.from_toml("router.toml")

    model_type, upstream = cfg.resolve("mirothinker-1.7")

    assert model_type == "mcp_first"
    assert upstream.base_url == "http://localhost:8000/v1"
