"""Git-based auto-update for llm_router."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class UpdateStatus(Enum):
    UP_TO_DATE = "up_to_date"
    BEHIND = "behind"
    ERROR = "error"


class UpdateResult(Enum):
    UPDATED = "updated"
    UP_TO_DATE = "up_to_date"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass
class UpdateInfo:
    status: UpdateStatus
    commits_behind: int = 0
    local_hash: str = ""
    remote_hash: str = ""
    error: str = ""


@dataclass
class UpdateDetail:
    result: UpdateResult
    old_hash: str = ""
    new_hash: str = ""
    commits_pulled: int = 0
    error: str = ""


def _run_git(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _find_uv() -> str | None:
    import shutil
    return shutil.which("uv")


class AutoUpdater:
    """Check for and apply git-based updates."""

    def __init__(self, project_root: Path, remote: str = "origin", branch: str = "main"):
        self.project_root = project_root
        self.remote = remote
        self.branch = branch
        self.available = self._check_available()

    def _check_available(self) -> bool:
        import shutil
        if not (self.project_root / ".git").exists():
            logger.info("No .git directory, auto-update unavailable")
            return False
        if not shutil.which("git"):
            logger.warning("git not found on PATH, auto-update unavailable")
            return False
        return True

    def check_for_updates(self) -> UpdateInfo:
        if not self.available:
            return UpdateInfo(status=UpdateStatus.ERROR, error="Auto-update unavailable")

        try:
            _run_git(["fetch", self.remote, "--prune"], self.project_root)

            local = _run_git(["rev-parse", "HEAD"], self.project_root)
            if local.returncode != 0:
                return UpdateInfo(status=UpdateStatus.ERROR, error="Cannot read local HEAD")

            remote = _run_git(["rev-parse", f"{self.remote}/{self.branch}"], self.project_root)
            if remote.returncode != 0:
                return UpdateInfo(
                    status=UpdateStatus.ERROR,
                    error=f"Remote branch {self.remote}/{self.branch} not found",
                )

            local_hash = local.stdout.strip()
            remote_hash = remote.stdout.strip()

            if local_hash == remote_hash:
                return UpdateInfo(
                    status=UpdateStatus.UP_TO_DATE,
                    local_hash=local_hash,
                    remote_hash=remote_hash,
                )

            count_result = _run_git(
                ["rev-list", "--count", f"{local_hash}..{remote_hash}"],
                self.project_root,
            )
            behind = int(count_result.stdout.strip()) if count_result.returncode == 0 else 0

            return UpdateInfo(
                status=UpdateStatus.BEHIND,
                commits_behind=behind,
                local_hash=local_hash,
                remote_hash=remote_hash,
            )

        except subprocess.TimeoutExpired:
            return UpdateInfo(status=UpdateStatus.ERROR, error="Git fetch timed out")
        except Exception as e:
            return UpdateInfo(status=UpdateStatus.ERROR, error=str(e))

    def apply_update(self) -> UpdateDetail:
        if not self.available:
            return UpdateDetail(result=UpdateResult.UNAVAILABLE, error="Auto-update unavailable")

        info = self.check_for_updates()
        if info.status != UpdateStatus.BEHIND:
            return UpdateDetail(
                result=UpdateResult.UP_TO_DATE,
                old_hash=info.local_hash,
            )

        old_hash = info.local_hash

        try:
            status = _run_git(["status", "--porcelain"], self.project_root)
            if status.stdout.strip():
                _run_git(["stash"], self.project_root)

            pull = _run_git(
                ["pull", "--ff-only", self.remote, self.branch],
                self.project_root,
            )
            if pull.returncode != 0:
                return UpdateDetail(
                    result=UpdateResult.FAILED,
                    old_hash=old_hash,
                    error=f"git pull failed: {pull.stderr.strip()}",
                )

            uv = _find_uv()
            if uv:
                sync = subprocess.run(
                    [uv, "sync"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if sync.returncode != 0:
                    logger.warning("uv sync failed: %s", sync.stderr.strip())

            new_hash_result = _run_git(["rev-parse", "HEAD"], self.project_root)
            new_hash = new_hash_result.stdout.strip() if new_hash_result.returncode == 0 else ""

            return UpdateDetail(
                result=UpdateResult.UPDATED,
                old_hash=old_hash,
                new_hash=new_hash,
                commits_pulled=info.commits_behind,
            )

        except subprocess.TimeoutExpired:
            return UpdateDetail(result=UpdateResult.FAILED, old_hash=old_hash, error="Timed out")
        except Exception as e:
            return UpdateDetail(result=UpdateResult.FAILED, old_hash=old_hash, error=str(e))

    def check_and_apply(self) -> UpdateDetail:
        info = self.check_for_updates()
        if info.status == UpdateStatus.UP_TO_DATE:
            return UpdateDetail(result=UpdateResult.UP_TO_DATE, old_hash=info.local_hash)
        if info.status == UpdateStatus.ERROR:
            return UpdateDetail(result=UpdateResult.FAILED, error=info.error)
        return self.apply_update()
