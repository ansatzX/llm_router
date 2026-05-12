"""Tests for llm-router CLI behavior."""

from types import SimpleNamespace

import llm_router.cli as cli


def test_debug_serve_disables_flask_reloader(monkeypatch, tmp_path):
    """Debug logging must not double-start the router process."""
    captured = {}
    cfg = SimpleNamespace(
        session_ttl_seconds=3600,
        server_host="127.0.0.1",
        server_port=0,
        upstreams={},
        routes=[],
    )

    monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(cli, "_load_config", lambda args: cfg)

    import llm_router.server as server_mod

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(server_mod.app, "run", fake_run)

    cli.cmd_serve(SimpleNamespace(config=None, debug=True))

    assert captured["debug"] is True
    assert captured["use_reloader"] is False
