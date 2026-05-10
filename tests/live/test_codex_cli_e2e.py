"""Live Codex CLI smoke tests.

These tests are intentionally opt-in. They start a temporary llm-router
process and drive the real Codex CLI against it, so they require local Codex
configuration and consume upstream model quota.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LLM_ROUTER_LIVE_CODEX_E2E") != "1",
    reason="set LLM_ROUTER_LIVE_CODEX_E2E=1 to run live Codex CLI e2e tests",
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_router_config(path: Path, port: int) -> None:
    path.write_text(
        f"""
[server]
host = "127.0.0.1"
port = {port}
session_ttl_seconds = 3600
max_rollback_retries = 3
mcp_server_name = "tools"

[upstream.default]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"

[upstream.deepseek]
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"

[[routes]]
pattern = "deepseek-v4-pro"
type = "responses_chat"
upstream = "deepseek"
upstream_model = "deepseek-reasoner"

[[routes]]
pattern = "deepseek-v4-flash"
type = "responses_chat"
upstream = "deepseek"
upstream_model = "deepseek-chat"

[[routes]]
pattern = "gpt-5.4-mini"
type = "responses_chat"
upstream = "deepseek"
upstream_model = "deepseek-chat"

[[routes]]
pattern = "gpt-5.4"
type = "responses_chat"
upstream = "deepseek"
upstream_model = "deepseek-reasoner"

[default_route]
type = "responses_chat"
upstream = "deepseek"
""".lstrip(),
        encoding="utf-8",
    )


def _wait_for_liveness(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 20
    url = f"http://127.0.0.1:{port}/liveness"
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(
                "llm-router exited before becoming live\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = exc
        time.sleep(0.2)

    raise AssertionError(f"llm-router did not become live: {last_error}")


def _terminate(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
        try:
            return process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.communicate(timeout=10)
    return process.communicate(timeout=1)


def test_codex_exec_can_complete_one_turn_through_temporary_router(tmp_path):
    """Real Codex exec should complete a simple prompt through llm-router."""
    if not shutil.which("codex"):
        pytest.skip("codex CLI is not on PATH")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required for live DeepSeek e2e")

    port = _free_port()
    router_config = tmp_path / "router.toml"
    router_home = tmp_path / "router-home"
    codex_cwd = tmp_path / "codex-workspace"
    _write_router_config(router_config, port)
    router_home.mkdir()
    codex_cwd.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(router_home)
    env["LLM_ROUTER_CONFIG"] = str(router_config)

    router = subprocess.Popen(
        ["uv", "run", "llm-router", "--config", str(router_config), "serve", "--debug"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_liveness(port, router)

        result = subprocess.run(
            [
                "codex",
                "exec",
                "-p",
                "llm_router",
                "-c",
                f'model_providers.llm_router.base_url="http://127.0.0.1:{port}/v1"',
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "-C",
                str(codex_cwd),
                "Say exactly: router-ok",
            ],
            cwd=codex_cwd,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            "codex exec failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "router-ok" in f"{result.stdout}\n{result.stderr}"

        debug_log = REPO_ROOT / "llm_router.jsonl"
        assert debug_log.exists()
        tail = debug_log.read_text(encoding="utf-8").splitlines()[-50:]
        assert any('"event": "CLIENT_REQUEST /v1/responses"' in line for line in tail)
    finally:
        stdout, stderr = _terminate(router)
        if router.returncode not in {0, -15}:
            raise AssertionError(
                "llm-router exited unexpectedly\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
