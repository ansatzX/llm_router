"""Tests for llm-router CLI behavior."""

from types import SimpleNamespace

import llm_router.cli as cli


def test_debug_serve_enables_flask_reloader(monkeypatch, tmp_path):
    """Debug serve should reload code changes during local development."""
    captured = {}
    cfg = SimpleNamespace(
        session_ttl_seconds=3600,
        server_host="127.0.0.1",
        server_port=0,
        upstreams={},
        routes=[],
    )

    monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(tmp_path / "sessions.json"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(cli, "_load_config", lambda args: cfg)

    import llm_router.server as server_mod

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(server_mod.app, "run", fake_run)

    cli.cmd_serve(SimpleNamespace(config=None, debug=True))

    assert captured["debug"] is True
    assert captured["use_reloader"] is True
    assert (tmp_path / "codex-home" / "llm_router.config.toml").exists()
    assert (tmp_path / "codex-home" / "llm_router.json").exists()


def test_clear_force_removes_corrupt_store_without_sessions(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Forced clear should delete corrupt artifacts even when no sessions load."""
    cfg = SimpleNamespace(
        session_ttl_seconds=3600,
        server_host="127.0.0.1",
        server_port=0,
        upstreams={},
        routes=[],
    )
    store_path = tmp_path / "sessions.json"
    store_path.write_text("{not valid json", encoding="utf-8")
    (tmp_path / "sessions.json.leftover.tmp").write_text("tmp", encoding="utf-8")

    monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(store_path))
    monkeypatch.setattr(cli, "_load_config", lambda args: cfg)

    cli.cmd_clear(SimpleNamespace(config=None, force=True))

    output = capsys.readouterr().out
    assert "Session files cleared" in output
    assert not list(tmp_path.glob("sessions.json*"))
