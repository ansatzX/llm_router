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


def test_route_can_override_upstream_model_name(tmp_path):
    """Routes may rewrite the model slug sent to the upstream provider."""
    config_path = tmp_path / "router.toml"
    config_path.write_text(
        """
[upstream.deepseek]
base_url = "https://api.deepseek.com"

[[routes]]
pattern = "gpt-*"
type = "responses"
upstream = "deepseek"
upstream_model = "deepseek-v4-flash"

[default_route]
type = "responses"
upstream = "deepseek"
""".strip(),
        encoding="utf-8",
    )

    cfg = RouterConfig.from_toml(config_path)

    model_type, upstream, upstream_model = cfg.resolve_request("gpt-5.4-mini")

    assert model_type == "responses"
    assert upstream.base_url == "https://api.deepseek.com"
    assert upstream_model == "deepseek-v4-flash"
