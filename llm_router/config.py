"""TOML configuration loader with model routing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import tomllib

DEFAULT_CONFIG_PATHS = [
    Path("router.toml"),
    Path.home() / ".config/llm-router/router.toml",
]


@dataclass
class UpstreamConfig:
    base_url: str
    api_key: str = ""
    api_key_env: str = ""

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclass
class RouteConfig:
    pattern: str
    model_type: str = "chat"  # "chat" | "mcp_first" | "responses"
    upstream: str = "default"

    def matches(self, model: str) -> bool:
        return fnmatch(model, self.pattern)


@dataclass
class RouterConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 9876
    default_model_type: str = "responses"
    default_upstream: str = "default"
    session_ttl_seconds: int = 3600
    max_rollback_retries: int = 3
    mcp_server_name: str = "tools"

    upstreams: dict[str, UpstreamConfig] = field(default_factory=dict)
    routes: list[RouteConfig] = field(default_factory=list)

    @classmethod
    def from_toml(cls, path: str | Path) -> RouterConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        cfg = cls()

        # [server]
        server = data.get("server", {})
        cfg.server_host = server.get("host", cfg.server_host)
        cfg.server_port = server.get("port", cfg.server_port)
        cfg.session_ttl_seconds = server.get("session_ttl_seconds", cfg.session_ttl_seconds)
        cfg.max_rollback_retries = server.get("max_rollback_retries", cfg.max_rollback_retries)
        cfg.mcp_server_name = server.get("mcp_server_name", cfg.mcp_server_name)

        # [upstream.*]
        for name, up_data in data.get("upstream", {}).items():
            cfg.upstreams[name] = UpstreamConfig(
                base_url=up_data.get("base_url", ""),
                api_key=up_data.get("api_key", ""),
                api_key_env=up_data.get("api_key_env", ""),
            )

        # [[routes]]
        for r_data in data.get("routes", []):
            cfg.routes.append(RouteConfig(
                pattern=r_data["pattern"],
                model_type=r_data.get("type", "chat"),
                upstream=r_data.get("upstream", "default"),
            ))

        # [default_route]
        default = data.get("default_route", {})
        cfg.default_model_type = default.get("type", cfg.default_model_type)
        cfg.default_upstream = default.get("upstream", cfg.default_upstream)

        cfg._validate()
        return cfg

    def _validate(self):
        if not self.upstreams:
            raise ValueError("At least one [upstream.*] must be defined")
        if self.default_upstream not in self.upstreams:
            raise ValueError(
                f"default_route.upstream '{self.default_upstream}' not in upstreams"
            )
        for route in self.routes:
            if route.upstream not in self.upstreams:
                raise ValueError(
                    f"Route '{route.pattern}' references unknown upstream '{route.upstream}'"
                )

    def resolve(self, model: str) -> tuple[str, UpstreamConfig]:
        """Resolve a model name to (model_type, upstream)."""
        for route in self.routes:
            if route.matches(model):
                upstream = self.upstreams[route.upstream]
                return route.model_type, upstream
        upstream = self.upstreams[self.default_upstream]
        return self.default_model_type, upstream

    @classmethod
    def load_or_find(cls, config_path: str | None = None) -> RouterConfig:
        """Load from explicit path, or search default locations."""
        if config_path:
            return cls.from_toml(config_path)
        return cls.find_and_load()

    @classmethod
    def find_and_load(cls) -> RouterConfig:
        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                return cls.from_toml(path)
        raise FileNotFoundError(
            f"No config found. Searched: {[str(p) for p in DEFAULT_CONFIG_PATHS]}"
        )
