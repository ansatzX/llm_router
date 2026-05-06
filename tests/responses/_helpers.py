"""Shared helpers for Responses endpoint tests."""

from unittest.mock import Mock

import llm_router.server as server_mod
from llm_router.config import RouterConfig, UpstreamConfig
from llm_router.session_store import SessionStore


def _configure_test_app(tmp_path, monkeypatch, llm_response):
    cfg = RouterConfig(
        upstreams={"default": UpstreamConfig(base_url="http://backend.test/v1")},
        routes=[],
        default_model_type="responses",
        default_upstream="default",
    )
    server_mod._config = cfg
    server_mod._sessions = SessionStore(
        store_path=tmp_path / "sessions.json",
        ttl_seconds=3600,
    )
    mock_make_request = Mock(return_value=llm_response)
    mock_make_responses_request = Mock()
    monkeypatch.setattr(server_mod, "make_llm_request", mock_make_request)
    monkeypatch.setattr(
        server_mod,
        "make_responses_request",
        mock_make_responses_request,
        raising=False,
    )
    server_mod.app.config.update(TESTING=True)
    return server_mod.app.test_client(), mock_make_request
