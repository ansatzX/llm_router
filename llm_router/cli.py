#!/usr/bin/env python3
"""CLI for the LLM Router server.

Commands:
    serve       Start the router server
    clear       Delete all persisted session history
    status      Show session store statistics
"""

import argparse
import os
import socket
import sys


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _load_config(args):
    """Shared config loading for CLI commands."""
    from llm_router.config import RouterConfig
    config_path = args.config or os.environ.get("LLM_ROUTER_CONFIG")
    return RouterConfig.load_or_find(config_path)


def _print_session_load_warning(stats):
    load_error = stats.get("load_error")
    if not load_error:
        return
    print(f"  WARNING: Session store could not be loaded: {load_error}")
    backup_path = stats.get("load_backup_path")
    if backup_path:
        print(f"  Preserved unreadable session file at: {backup_path}")


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def _port_available(host: str, port: int) -> bool:
    if port == 0:
        return True
    try:
        with socket.create_server((host, port), reuse_port=False):
            return True
    except OSError:
        return False


def _exit_port_in_use(host: str, port: int) -> None:
    print(
        f"Port {port} on {host} is already in use. "
        "Please start llm-router with a temporary port, for example "
        f"`llm-router serve --port {port + 1}`, and update the matching "
        "`base_url` port in `~/.codex/config.toml`.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _is_werkzeug_reloader_child() -> bool:
    return os.environ.get("WERKZEUG_RUN_MAIN") == "true"


def cmd_serve(args):
    """Start the router server."""
    _load_dotenv()

    from llm_router.debug_log import set_debug_mode
    from llm_router.session_store import SessionStore

    cfg = _load_config(args)
    if args.port is not None:
        cfg.server_port = args.port
    is_reloader_child = _is_werkzeug_reloader_child()
    if not is_reloader_child and not _port_available(cfg.server_host, cfg.server_port):
        _exit_port_in_use(cfg.server_host, cfg.server_port)

    from llm_router.codex_presets import sync_codex_presets

    synced_presets = [] if is_reloader_child else sync_codex_presets()

    # Wire globals
    import llm_router.server as server_mod
    server_mod._config = cfg
    server_mod._sessions = SessionStore(ttl_seconds=cfg.session_ttl_seconds)

    if args.debug:
        set_debug_mode(True)
        print("[DEBUG] Debug mode enabled, logging to llm_router.jsonl")

    print(f"LLM Router starting on {cfg.server_host}:{cfg.server_port}")
    print(f"  Upstreams: {list(cfg.upstreams.keys())}")
    print(f"  Routes: {len(cfg.routes)} patterns")
    if synced_presets:
        print(f"  Codex presets: {len(synced_presets)} synced to {synced_presets[0].parent}")
    print(f"  Sessions: {len(server_mod._sessions)} loaded from disk")
    _print_session_load_warning(server_mod._sessions.stats())

    from llm_router.server import app
    app.run(
        host=cfg.server_host,
        port=cfg.server_port,
        debug=args.debug,
        use_reloader=args.debug,
    )


def cmd_clear(args):
    """Delete all persisted session history with confirmation."""
    _load_dotenv()

    from llm_router.session_store import SessionStore
    cfg = _load_config(args)
    store = SessionStore(ttl_seconds=cfg.session_ttl_seconds)
    stats = store.stats()

    if stats["session_count"] == 0 and stats["clear_file_count"] == 0:
        _print_session_load_warning(stats)
        print("No sessions to clear.")
        return

    print("=" * 60)
    print("  WARNING: You are about to delete ALL session history.")
    print()
    print(f"  Sessions:    {stats['session_count']}")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Stored at:   {stats['store_path']}")
    print(f"  Disk size:   {stats['store_size_bytes']:,} bytes")
    print(f"  Store files: {stats['clear_file_count']}")
    print(f"  File bytes:  {stats['clear_file_size_bytes']:,} bytes")
    _print_session_load_warning(stats)
    print()
    print("  This removes persisted conversation state plus")
    print("  leftover corrupt or temporary session-store files.")
    print()
    print("  This action CANNOT be undone.")
    print("=" * 60)

    if not args.force:
        print()
        response = input("  Type 'DELETE ALL SESSIONS' to confirm: ")
        if response.strip() != "DELETE ALL SESSIONS":
            print("  Aborted. No sessions were deleted.")
            return

    count = store.clear_all()
    print(f"\n  Deleted {count} sessions. Session files cleared.")


def cmd_status(args):
    """Show session store statistics."""
    _load_dotenv()

    from llm_router.session_store import SessionStore
    cfg = _load_config(args)
    store = SessionStore(ttl_seconds=cfg.session_ttl_seconds)
    stats = store.stats()

    print("Session Store")
    print(f"  Path:      {stats['store_path']}")
    print(f"  Sessions:  {stats['session_count']}")
    print(f"  Items:     {stats['total_items']}")
    print(f"  Disk size: {stats['store_size_bytes']:,} bytes")
    print(f"  TTL:       {cfg.session_ttl_seconds}s")
    _print_session_load_warning(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Router — API proxy with session persistence",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to router.toml",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start the router server")
    p_serve.add_argument("--port", type=_parse_port, default=None,
                         help="Override the server port for this run")
    p_serve.add_argument("--debug", action="store_true",
                         help="Enable debug logging")
    p_serve.set_defaults(func=cmd_serve)

    # clear
    p_clear = sub.add_parser("clear", help="Delete all session history")
    p_clear.add_argument("--force", "-f", action="store_true",
                         help="Skip confirmation prompt")
    p_clear.set_defaults(func=cmd_clear)

    # status
    p_status = sub.add_parser("status", help="Show session statistics")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
