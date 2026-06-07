"""Manage the llm_router Flask server as a subprocess."""

from __future__ import annotations

import collections
import logging
import socket
import subprocess
import sys
import threading
import time

logger = logging.getLogger(__name__)

_MAX_LOG_LINES = 500
_MAX_CRASH_RESTARTS = 3
_CRASH_BACKOFF_SECONDS = 5


def _find_uv() -> str | None:
    """Locate the uv executable on PATH."""
    import shutil
    return shutil.which("uv")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


class ServerManager:
    """Start, stop, and monitor the llm_router server subprocess."""

    def __init__(
        self,
        config_path: str | None = None,
        host: str = "127.0.0.1",
        port: int = 9876,
        debug: bool = False,
    ):
        self.config_path = config_path
        self.host = host
        self.port = port
        self.debug = debug
        self._process: subprocess.Popen | None = None
        self._log_lines: collections.deque[str] = collections.deque(maxlen=_MAX_LOG_LINES)
        self._lock = threading.Lock()
        self._stdout_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._crash_restarts: int = 0
        self._on_state_change: list = []
        self._stopping = False

    def on_state_change(self, callback) -> None:
        self._on_state_change.append(callback)

    def _notify(self, state: str) -> None:
        for cb in self._on_state_change:
            try:
                cb(state)
            except Exception:
                logger.exception("State change callback failed")

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self) -> dict:
        running = self.is_running()
        return {
            "running": running,
            "pid": self._process.pid if running else None,
            "uptime": time.time() - self._start_time if running else 0,
            "returncode": self._process.returncode if self._process else None,
        }

    def get_logs(self, n: int = 100) -> list[str]:
        with self._lock:
            return list(self._log_lines)[-n:]

    def start(self) -> None:
        if self.is_running():
            logger.warning("Server already running (pid %d)", self._process.pid)
            return

        if _port_in_use(self.host, self.port):
            raise RuntimeError(
                f"Port {self.port} is already in use. "
                "Another instance may be running."
            )

        cmd = self._build_command()
        logger.info("Starting server: %s", cmd)

        self._stopping = False
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        self._start_time = time.time()
        self._notify("started")

        self._stdout_thread = threading.Thread(
            target=self._read_output, daemon=True
        )
        self._stdout_thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog, daemon=True
        )
        self._watchdog_thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if not self.is_running():
            return

        self._stopping = True
        logger.info("Stopping server (pid %d)", self._process.pid)

        try:
            self._process.terminate()
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully, killing")
            self._process.kill()
            self._process.wait(timeout=5)
        except Exception:
            logger.exception("Error stopping server")

        self._notify("stopped")

    def restart(self) -> None:
        self.stop()
        self._crash_restarts = 0
        self.start()

    def _build_command(self) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "serve"]

        uv = _find_uv()
        if uv:
            cmd = [uv, "run", "llm-router", "serve"]
        else:
            cmd = [sys.executable, "-m", "llm_router.cli", "serve"]

        if self.config_path:
            cmd.extend(["--config", self.config_path])
        if self.debug:
            cmd.append("--debug")
        return cmd

    def _read_output(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            line = line.rstrip("\n\r")
            with self._lock:
                self._log_lines.append(line)
        # Process ended
        self._process.stdout.close()

    def _watchdog(self) -> None:
        assert self._process is not None
        self._process.wait()
        rc = self._process.returncode
        if self._stopping:
            self._notify("stopped")
            return

        if rc != 0:
            logger.error("Server crashed with exit code %d", rc)
            self._notify("crashed")

            if self._crash_restarts < _MAX_CRASH_RESTARTS:
                self._crash_restarts += 1
                logger.info(
                    "Auto-restarting in %d seconds (attempt %d/%d)",
                    _CRASH_BACKOFF_SECONDS,
                    self._crash_restarts,
                    _MAX_CRASH_RESTARTS,
                )
                time.sleep(_CRASH_BACKOFF_SECONDS)
                if not self._stopping:
                    try:
                        self.start()
                    except Exception:
                        logger.exception("Auto-restart failed")
            else:
                logger.error("Max crash restarts reached")
                self._notify("error")
        else:
            self._notify("stopped")
