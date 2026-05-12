"""Chat Completions endpoint regressions."""

import llm_router.server as server_mod
from llm_router.config import RouterConfig, UpstreamConfig
from llm_router.session_store import SessionStore


def test_chat_completions_forwards_tools_to_upstream(tmp_path, monkeypatch):
    """The stateless chat endpoint must not silently drop provider tools."""
    cfg = RouterConfig(
        upstreams={"xiaomi": UpstreamConfig(base_url="https://api.xiaomimimo.com")},
        routes=[],
        default_model_type="responses_chat",
        default_upstream="xiaomi",
    )
    server_mod._config = cfg
    server_mod._sessions = SessionStore(
        store_path=tmp_path / "sessions.json",
        ttl_seconds=3600,
    )
    captured = {}

    def fake_run(payload, *_args, **_kwargs):
        captured["payload"] = payload
        return (
            {
                "created": 123,
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            None,
            0,
            "ok",
        )

    monkeypatch.setattr(server_mod, "_run_llm_with_rollback", fake_run)
    server_mod.app.config.update(TESTING=True)
    client = server_mod.app.test_client()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-pro",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search", "force_search": True}],
            "thinking": {"type": "disabled"},
        },
    )

    assert response.status_code == 200
    assert captured["payload"]["tools"] == [
        {"type": "web_search", "force_search": True}
    ]
