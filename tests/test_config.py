"""Tests for TOML router configuration."""

from llm_router.config import RouterConfig


def test_deepseek_models_route_as_responses_chat():
    """DeepSeek routes through stateful Responses with Chat upstream adapter."""
    cfg = RouterConfig.from_toml("router.toml")

    model_type, upstream = cfg.resolve("deepseek-chat")

    assert model_type == "responses_chat"
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


def test_route_can_select_explicit_responses_passthrough_provider(tmp_path):
    """Native Responses passthrough is an explicit route behavior."""
    config_path = tmp_path / "router.toml"
    config_path.write_text(
        """
[upstream.deepseek]
base_url = "https://api.deepseek.com"

[upstream.aicc0]
base_url = "https://zapi.aicc0.com/v1"

[[routes]]
pattern = "deepseek-gateway-*"
type = "responses_passthrough"
upstream = "aicc0"
upstream_model = "deepseek-v4-pro"

[[routes]]
pattern = "deepseek-*"
type = "responses_chat"
upstream = "deepseek"

[default_route]
type = "responses"
upstream = "deepseek"
""".strip(),
        encoding="utf-8",
    )

    cfg = RouterConfig.from_toml(config_path)

    gateway_type, gateway_upstream, gateway_model = cfg.resolve_request(
        "deepseek-gateway-pro",
    )
    official_type, official_upstream, official_model = cfg.resolve_request(
        "deepseek-chat",
    )

    assert gateway_type == "responses_passthrough"
    assert gateway_upstream.base_url == "https://zapi.aicc0.com/v1"
    assert gateway_model == "deepseek-v4-pro"
    assert official_type == "responses_chat"
    assert official_upstream.base_url == "https://api.deepseek.com"
    assert official_model == "deepseek-chat"


def test_config_rejects_unknown_route_type(tmp_path):
    """Route type typos should fail at load time."""
    config_path = tmp_path / "router.toml"
    config_path.write_text(
        """
[upstream.default]
base_url = "https://backend.test/v1"

[[routes]]
pattern = "*"
type = "resposne"
upstream = "default"

[default_route]
type = "responses"
upstream = "default"
""".strip(),
        encoding="utf-8",
    )

    try:
        RouterConfig.from_toml(config_path)
    except ValueError as exc:
        assert "unknown route type" in str(exc)
    else:
        raise AssertionError("unknown route type should fail validation")
