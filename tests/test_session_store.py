"""Tests for persisted Responses API session state."""

from llm_router.session_store import SessionStore


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
