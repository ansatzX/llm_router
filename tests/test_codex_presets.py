"""Tests for synchronizing Codex preset files."""

from pathlib import Path

from llm_router.codex_presets import discover_preset_files, sync_codex_presets


def test_discover_preset_files_only_selects_root_presets(tmp_path):
    """Preset discovery should avoid logs, examples, and nested package data."""
    (tmp_path / "llm_router.config.toml").write_text("model = 'x'", encoding="utf-8")
    (tmp_path / "llm_router.json").write_text("{}", encoding="utf-8")
    (tmp_path / "codex.config.example.toml").write_text("example", encoding="utf-8")
    (tmp_path / "llm_router.jsonl").write_text("log", encoding="utf-8")
    nested = tmp_path / "llm_router"
    nested.mkdir()
    (nested / "quotes.json").write_text("{}", encoding="utf-8")

    files = discover_preset_files(tmp_path)

    assert [path.name for path in files] == [
        "llm_router.config.toml",
        "llm_router.json",
    ]


def test_sync_codex_presets_copies_and_overwrites_files(tmp_path):
    """Sync should create CODEX_HOME and overwrite stale preset contents."""
    source_dir = tmp_path / "source"
    codex_home = tmp_path / "codex-home"
    source_dir.mkdir()
    (source_dir / "aihubmix.config.toml").write_text("model = 'new'", encoding="utf-8")
    (source_dir / "aihubmix.json").write_text('{"new": true}', encoding="utf-8")
    codex_home.mkdir()
    (codex_home / "aihubmix.config.toml").write_text("old", encoding="utf-8")

    copied = sync_codex_presets(source_dir=source_dir, codex_home=codex_home)

    assert [path.name for path in copied] == [
        "aihubmix.config.toml",
        "aihubmix.json",
    ]
    assert (codex_home / "aihubmix.config.toml").read_text(encoding="utf-8") == (
        "model = 'new'"
    )
    assert (codex_home / "aihubmix.json").read_text(encoding="utf-8") == (
        '{"new": true}'
    )


def test_sync_codex_presets_uses_codex_home_env(tmp_path, monkeypatch):
    """CODEX_HOME should select the target directory on every platform."""
    source_dir = tmp_path / "source"
    codex_home = tmp_path / "custom-codex-home"
    source_dir.mkdir()
    (source_dir / "llm_router.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    copied = sync_codex_presets(source_dir=source_dir)

    assert copied == [Path(codex_home / "llm_router.json")]
    assert (codex_home / "llm_router.json").read_text(encoding="utf-8") == "{}"
