"""Pytest configuration and fixtures for parser tests."""
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_llm_response():
    """Create standard mock LLM response for tests."""
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "id": "test",
        "choices": [{"message": {"content": "test"}}]
    }
    return mock_response


@pytest.fixture
def reset_llm_client_state():
    """Reset llm_client global state before and after each test.

    This fixture resets both _REQUEST_TIMEOUT and _CLIENT_CACHE to ensure
    test isolation. Used by tests that modify global client state.
    """
    import llm_router.llm_client as client_module

    # Reset before test
    client_module._REQUEST_TIMEOUT = None
    client_module._CLIENT_CACHE.clear()

    yield

    # Reset after test
    client_module._REQUEST_TIMEOUT = None
    client_module._CLIENT_CACHE.clear()
