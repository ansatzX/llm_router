"""Tests for llm-router CLI behavior."""

import socket
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

    cli.cmd_serve(SimpleNamespace(config=None, debug=True, port=None))

    assert captured["debug"] is True
    assert captured["use_reloader"] is True
    assert (tmp_path / "codex-home" / "llm_router.config.toml").exists()
    assert (tmp_path / "codex-home" / "llm_router.json").exists()


def test_serve_port_override_updates_run_port(monkeypatch, tmp_path):
    """--port should override the configured server port for this run."""
    captured = {}
    cfg = SimpleNamespace(
        session_ttl_seconds=3600,
        server_host="127.0.0.1",
        server_port=9876,
        upstreams={},
        routes=[],
    )

    monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(tmp_path / "sessions.json"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(cli, "_load_config", lambda args: cfg)

    import llm_router.server as server_mod

    monkeypatch.setattr(server_mod.app, "run", lambda **kwargs: captured.update(kwargs))

    cli.cmd_serve(SimpleNamespace(config=None, debug=False, port=9877))

    assert captured["port"] == 9877
    assert cfg.server_port == 9877


def test_serve_exits_before_side_effects_when_port_is_in_use(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Occupied ports should fail before syncing Codex presets or sessions."""
    cfg = SimpleNamespace(
        session_ttl_seconds=3600,
        server_host="127.0.0.1",
        server_port=0,
        upstreams={},
        routes=[],
    )
    codex_home = tmp_path / "codex-home"
    session_path = tmp_path / "sessions.json"

    with socket.create_server((cfg.server_host, 0)) as listener:
        cfg.server_port = listener.getsockname()[1]
        monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(session_path))
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        monkeypatch.setattr(cli, "_load_config", lambda args: cfg)

        try:
            cli.cmd_serve(SimpleNamespace(config=None, debug=False, port=None))
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - assertion reports details
            raise AssertionError("cmd_serve should exit when the port is occupied")

    output = capsys.readouterr()
    assert "already in use" in output.err
    assert "--port" in output.err
    assert "~/.codex/config.toml" in output.err
    assert not codex_home.exists()
    assert not session_path.exists()


def test_debug_reloader_child_skips_port_check_and_preset_sync(
    monkeypatch,
    tmp_path,
):
    """Werkzeug reloader child inherits the server socket, so it must not pre-bind."""
    captured = {}
    cfg = SimpleNamespace(
        session_ttl_seconds=3600,
        server_host="127.0.0.1",
        server_port=0,
        upstreams={},
        routes=[],
    )
    codex_home = tmp_path / "codex-home"

    with socket.create_server((cfg.server_host, 0)) as listener:
        cfg.server_port = listener.getsockname()[1]
        monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
        monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(tmp_path / "sessions.json"))
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        monkeypatch.setattr(cli, "_load_config", lambda args: cfg)

        import llm_router.server as server_mod

        monkeypatch.setattr(server_mod.app, "run", lambda **kwargs: captured.update(kwargs))

        cli.cmd_serve(SimpleNamespace(config=None, debug=True, port=None))

    assert captured["port"] == cfg.server_port
    assert captured["use_reloader"] is True
    assert not codex_home.exists()


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
