"""System tray application for llm_router on Windows."""

from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

from llm_router.windows.server_manager import ServerManager
from llm_router.windows.updater import AutoUpdater, UpdateResult

logger = logging.getLogger(__name__)


def _create_icon(color: str, size: int = 64):
    """Generate a simple circle icon with PIL."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    colors = {
        "green": "#00e676",
        "red": "#ff5252",
        "yellow": "#ffd740",
    }
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=colors.get(color, colors["red"]))
    return img


def _get_project_root() -> Path:
    """Find the project root (directory containing router.toml)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent


class TrayApp:
    """System tray icon and menu for managing llm_router."""

    def __init__(self, server_manager: ServerManager, config):
        self.manager = server_manager
        self.config = config
        self._icon = None
        self._updater: AutoUpdater | None = None
        self._update_thread: threading.Thread | None = None

    def run(self) -> None:
        """Start the tray application (blocking)."""
        try:
            import pystray
            from pystray import Menu, MenuItem
        except ImportError:
            print(
                "pystray is required for the tray application.\n"
                "Install with: uv sync --extra windows",
                file=sys.stderr,
            )
            sys.exit(1)

        root = _get_project_root()
        self._updater = AutoUpdater(
            project_root=root,
            remote=self.config.git_remote,
            branch=self.config.git_branch,
        )

        self.manager.on_state_change(self._on_server_state_change)

        icon_image = _create_icon("red")

        menu = Menu(
            MenuItem("Open Dashboard", self._open_dashboard, default=True),
            Menu.SEPARATOR,
            MenuItem(self._status_text, None, enabled=False),
            MenuItem("Start Server", self._start_server, visible=lambda item: not self.manager.is_running()),
            MenuItem("Stop Server", self._stop_server, visible=lambda item: self.manager.is_running()),
            MenuItem("Restart Server", self._restart_server, visible=lambda item: self.manager.is_running()),
            Menu.SEPARATOR,
            MenuItem("Check for Updates", self._check_updates),
            MenuItem(
                "Auto-Update",
                self._toggle_auto_update,
                checked=lambda item: self.config.auto_update_enabled,
            ),
            Menu.SEPARATOR,
            MenuItem("View Debug Log", self._open_debug_log),
            MenuItem("Clear Sessions", self._clear_sessions),
            Menu.SEPARATOR,
            MenuItem("Settings", self._open_settings),
            MenuItem("About", self._show_about),
            Menu.SEPARATOR,
            MenuItem("Quit", self._quit),
        )

        self._icon = pystray.Icon(
            "llm-router",
            icon_image,
            "LLM Router",
            menu,
        )

        if self.config.auto_start_server:
            threading.Thread(target=self._auto_start, daemon=True).start()

        if self.config.auto_update_enabled:
            self._start_update_loop()

        self._icon.run()

    def _status_text(self, _item=None) -> str:
        st = self.manager.status()
        if st["running"]:
            return f"Server: Running (PID {st['pid']})"
        return "Server: Stopped"

    def _on_server_state_change(self, state: str) -> None:
        if self._icon is None:
            return
        color = {"started": "green", "stopped": "red", "crashed": "yellow", "error": "red"}.get(state, "red")
        new_img = _create_icon(color)
        if new_img:
            self._icon.icon = new_img
        self._icon.update_menu()

    def _auto_start(self) -> None:
        import time
        time.sleep(0.5)
        try:
            self.manager.start()
            self._notify("Server started")
        except Exception as e:
            logger.exception("Auto-start failed")
            self._notify(f"Start failed: {e}")

    def _start_server(self, _item=None) -> None:
        try:
            self.manager.start()
            self._notify("Server started")
        except Exception as e:
            self._notify(f"Start failed: {e}")

    def _stop_server(self, _item=None) -> None:
        self.manager.stop()
        self._notify("Server stopped")

    def _restart_server(self, _item=None) -> None:
        self.manager.restart()
        self._notify("Server restarted")

    def _open_dashboard(self, _item=None) -> None:
        host = self.manager.host
        port = self.manager.port
        webbrowser.open(f"http://{host}:{port}/dashboard")

    def _check_updates(self, _item=None) -> None:
        if self._updater is None or not self._updater.available:
            self._notify("Auto-update unavailable (no .git or no git)")
            return

        def _do_check():
            detail = self._updater.check_and_apply()
            if detail.result == UpdateResult.UPDATED:
                self._notify(f"Updated! {detail.commits_pulled} commits pulled. Restarting...")
                self.manager.restart()
            elif detail.result == UpdateResult.UP_TO_DATE:
                self._notify("Already up to date")
            else:
                self._notify(f"Update failed: {detail.error}")

        threading.Thread(target=_do_check, daemon=True).start()

    def _toggle_auto_update(self, _item=None) -> None:
        self.config.auto_update_enabled = not self.config.auto_update_enabled

    def _open_debug_log(self, _item=None) -> None:
        log_path = Path("llm_router.jsonl")
        if log_path.exists():
            os.startfile(str(log_path))
        else:
            self._notify("No debug log found")

    def _clear_sessions(self, _item=None) -> None:
        try:
            from llm_router.server import _config
            from llm_router.session_store import SessionStore
            if _config is None:
                self._notify("No config loaded")
                return
            store = SessionStore(ttl_seconds=_config.session_ttl_seconds)
            stats = store.stats()
            if stats["session_count"] == 0:
                self._notify("No sessions to clear")
                return
            count = store.clear_all()
            self._notify(f"Deleted {count} sessions")
        except Exception as e:
            self._notify(f"Clear failed: {e}")

    def _open_settings(self, _item=None) -> None:
        config_path = Path("router.toml")
        if config_path.exists():
            os.startfile(str(config_path))
        else:
            self._notify("No router.toml found")

    def _show_about(self, _item=None) -> None:
        from llm_router import __version__
        self._notify(f"LLM Router v{__version__}\nhttps://github.com/ansatzX/llm_router")

    def _quit(self, _item=None) -> None:
        self.manager.stop()
        if self._icon:
            self._icon.stop()

    def _notify(self, message: str) -> None:
        if self._icon:
            try:
                self._icon.notify(message, "LLM Router")
            except Exception:
                logger.debug("Notification failed", exc_info=True)

    def _start_update_loop(self) -> None:
        if self._updater is None or not self._updater.available:
            return

        def _loop():
            import time
            while True:
                time.sleep(self.config.check_update_interval)
                if not self.config.auto_update_enabled:
                    continue
                try:
                    info = self._updater.check_for_updates()
                    from llm_router.windows.updater import UpdateStatus
                    if info.status == UpdateStatus.BEHIND:
                        self._notify(f"Update available ({info.commits_behind} commits behind)")
                        detail = self._updater.apply_update()
                        if detail.result == UpdateResult.UPDATED:
                            self._notify(f"Updated to {detail.new_hash[:8]}. Restarting...")
                            self.manager.restart()
                except Exception:
                    logger.exception("Auto-update check failed")

        self._update_thread = threading.Thread(target=_loop, daemon=True)
        self._update_thread.start()
