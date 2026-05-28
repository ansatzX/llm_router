"""Tests for persisted Responses API session state."""

import json
import threading

import llm_router.session_store as session_store_mod
from llm_router.session_store import SessionStore


def test_default_store_path_uses_startup_directory(tmp_path, monkeypatch):
    """Default session state should stay local to the router process cwd."""
    monkeypatch.delenv("LLM_ROUTER_SESSION_STORE", raising=False)
    monkeypatch.chdir(tmp_path)

    store = SessionStore(ttl_seconds=3600)

    assert store.store_path == tmp_path / ".llm-router" / "sessions.json"


def test_store_path_can_be_overridden_by_environment(tmp_path, monkeypatch):
    """Deployments can opt into an explicit durable session location."""
    store_path = tmp_path / "state" / "sessions.json"
    monkeypatch.setenv("LLM_ROUTER_SESSION_STORE", str(store_path))

    store = SessionStore(ttl_seconds=3600)
    session = store.create("test-model")

    assert store.store_path == store_path
    assert store_path.exists()
    assert store.get(session.response_id) is session


def test_session_mutations_are_persisted(tmp_path):
    """New user and assistant items survive store reloads."""
    store_path = tmp_path / "sessions.json"
    store = SessionStore(store_path=store_path, ttl_seconds=3600)

    session = store.create("test-model")
    store.add_items(
        session,
        [{"type": "message", "role": "user", "content": "Hello"}],
    )
    store.add_output_item(
        session,
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hi"}],
        },
    )

    reloaded = SessionStore(store_path=store_path, ttl_seconds=3600)
    same_session = reloaded.get(session.response_id)

    assert same_session is not None
    assert same_session.items == session.items


def test_malformed_json_store_is_preserved_and_ignored(tmp_path):
    """Malformed persisted state should not prevent router startup."""
    store_path = tmp_path / "sessions.json"
    store_path.write_text("{not valid json", encoding="utf-8")

    store = SessionStore(store_path=store_path, ttl_seconds=3600)
    stats = store.stats()

    assert len(store) == 0
    assert not store_path.exists()
    assert stats["load_error"].startswith("JSONDecodeError:")
    backup_path = stats["load_backup_path"]
    assert backup_path is not None
    assert store.load_backup_path is not None
    assert str(store.load_backup_path) == backup_path
    assert store.load_backup_path.read_text(encoding="utf-8") == "{not valid json"

    session = store.create("test-model")
    assert store.get(session.response_id) is session
    assert store_path.exists()


def test_non_utf8_store_is_preserved_and_ignored(tmp_path):
    """Invalid UTF-8 in persisted state should not crash SessionStore construction."""
    store_path = tmp_path / "sessions.json"
    corrupt_bytes = b'{"sessions": {"resp_bad": "\xf0"}}'
    store_path.write_bytes(corrupt_bytes)

    store = SessionStore(store_path=store_path, ttl_seconds=3600)
    stats = store.stats()

    assert len(store) == 0
    assert stats["load_error"].startswith("UnicodeDecodeError:")
    backup_path = stats["load_backup_path"]
    assert backup_path is not None
    assert store.load_backup_path is not None
    assert str(store.load_backup_path) == backup_path
    assert store.load_backup_path.read_bytes() == corrupt_bytes


def test_structurally_corrupt_store_does_not_partially_load_sessions(tmp_path):
    """A bad later session should not leave earlier sessions live in memory."""
    store_path = tmp_path / "sessions.json"
    original = SessionStore(store_path=store_path, ttl_seconds=3600)
    session = original.create("test-model")

    data = json.loads(store_path.read_text(encoding="utf-8"))
    data["sessions"]["resp_bad"] = {"model": "broken"}
    store_path.write_text(json.dumps(data), encoding="utf-8")

    store = SessionStore(store_path=store_path, ttl_seconds=3600)
    stats = store.stats()

    assert len(store) == 0
    assert store.get(session.response_id) is None
    assert not store_path.exists()
    assert stats["load_error"].startswith("KeyError:")
    assert store.load_backup_path is not None


def test_clear_all_removes_corrupt_and_temporary_session_files(tmp_path):
    """Clear should remove stale session artifacts, not only live JSON state."""
    store_path = tmp_path / "sessions.json"
    store_path.write_text("{not valid json", encoding="utf-8")
    tmp_file = tmp_path / "sessions.json.abc.tmp"
    old_tmp_file = tmp_path / "sessions.json.tmp"
    old_corrupt_file = tmp_path / "sessions.json.corrupt-old"
    tmp_file.write_text("tmp", encoding="utf-8")
    old_tmp_file.write_text("tmp", encoding="utf-8")
    old_corrupt_file.write_text("old", encoding="utf-8")

    store = SessionStore(store_path=store_path, ttl_seconds=3600)

    assert store.stats()["clear_file_count"] == 4
    assert store.clear_all() == 0
    assert not list(tmp_path.glob("sessions.json*"))


def test_register_response_id_keeps_latest_response_link(tmp_path):
    """A new response ID can be used as previous_response_id on the next turn."""
    store = SessionStore(store_path=tmp_path / "sessions.json", ttl_seconds=3600)
    session = store.create("test-model")

    store.register_response_id(session, "resp_next")

    assert store.get("resp_next") is session
    assert session.response_id == "resp_next"


def test_concurrent_store_instances_do_not_share_temp_file(tmp_path, monkeypatch):
    """Independent router processes should not race on one fixed temp path."""
    store_path = tmp_path / "sessions.json"
    monkeypatch.setattr(session_store_mod, "fcntl", None)
    stores = [
        SessionStore(store_path=store_path, ttl_seconds=3600),
        SessionStore(store_path=store_path, ttl_seconds=3600),
    ]
    original_dump = json.dump
    barrier = threading.Barrier(2)

    def synchronized_dump(*args, **kwargs):
        original_dump(*args, **kwargs)
        barrier.wait(timeout=5)

    monkeypatch.setattr(json, "dump", synchronized_dump)

    errors = []

    def create_session(store):
        try:
            store.create("test-model")
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    threads = [threading.Thread(target=create_session, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
