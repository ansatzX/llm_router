"""Windows-specific configuration for the system tray wrapper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindowsConfig:
    auto_update_enabled: bool = True
    git_remote: str = "origin"
    git_branch: str = "main"
    auto_start_server: bool = True
    check_update_interval: int = 3600

    @classmethod
    def from_router_config(cls, cfg) -> WindowsConfig:
        """Load from a RouterConfig's [windows] section."""
        raw = getattr(cfg, "windows", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            auto_update_enabled=raw.get("auto_update_enabled", True),
            git_remote=raw.get("git_remote", "origin"),
            git_branch=raw.get("git_branch", "main"),
            auto_start_server=raw.get("auto_start_server", True),
            check_update_interval=raw.get("check_update_interval", 3600),
        )
