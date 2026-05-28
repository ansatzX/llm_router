"""Synchronize bundled Codex profile and model catalog presets."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

PRESET_PATTERNS = ("*.config.toml", "*.json")
CODEX_HOME_ENV = "CODEX_HOME"


def default_codex_home() -> Path:
    """Return Codex's user state directory."""
    configured = os.environ.get(CODEX_HOME_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def default_preset_source_dir() -> Path:
    """Return the project root that contains router Codex presets."""
    return Path(__file__).resolve().parent.parent


def discover_preset_files(source_dir: Path | None = None) -> list[Path]:
    """Find root-level Codex preset files to copy into CODEX_HOME."""
    root = source_dir or default_preset_source_dir()
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in PRESET_PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            files.append(path)
    return sorted(files, key=lambda path: path.name)


def _copy_overwrite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            fd = -1
            tmp_file.write(source.read_bytes())
        os.replace(tmp_path, destination)
    except Exception:
        if fd != -1:
            os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise


def sync_codex_presets(
    *,
    source_dir: Path | None = None,
    codex_home: Path | None = None,
) -> list[Path]:
    """Copy router Codex presets into CODEX_HOME and overwrite existing files."""
    target_dir = codex_home or default_codex_home()
    copied: list[Path] = []
    for source in discover_preset_files(source_dir):
        destination = target_dir / source.name
        _copy_overwrite(source, destination)
        copied.append(destination)
    return copied
