"""Tests for structured debug logging."""

import json

import llm_router.debug_log as debug_log


def test_log_debug_writes_jsonl_and_truncates_nested_payload(tmp_path, monkeypatch):
    """Debug logging should emit one JSON object per line."""
    log_path = tmp_path / "llm_router.jsonl"
    monkeypatch.setattr(debug_log, "DEBUG_LOG_FILE", str(log_path))

    debug_log.set_debug_mode(True)
    try:
        debug_log.log_debug(
            "TEST_EVENT",
            {
                "payload": {
                    "content": "x" * 600,
                    "nested": [{"text": "y" * 550}],
                },
            },
        )
    finally:
        debug_log.set_debug_mode(False)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["event"] == "TEST_EVENT"
    assert "ts" in entry
    assert entry["data"]["payload"]["content"].endswith(
        "... (truncated, 600 total chars)"
    )
    assert entry["data"]["payload"]["nested"][0]["text"].endswith(
        "... (truncated, 550 total chars)"
    )
