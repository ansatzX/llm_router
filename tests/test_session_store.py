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
