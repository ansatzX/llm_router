"""Tests for default parameter handling in llm_client."""

import os
from unittest.mock import Mock, patch

import pytest

from llm_router.llm_client import LLMRequestError, make_llm_request


def test_missing_sampling_params_are_not_injected(mock_llm_response):
    """Transport must not add provider parameters the client did not request."""
    with patch('llm_router.llm_client._get_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_llm_response
        mock_get_client.return_value = mock_client

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        make_llm_request(payload, "http://localhost:8000", None)

        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert "temperature" not in call_kwargs
        assert "top_p" not in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert "extra_body" not in call_kwargs


def test_client_params_override_defaults(mock_llm_response):
    """Test that client-specified parameters override defaults."""
    with patch('llm_router.llm_client._get_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_llm_response
        mock_get_client.return_value = mock_client

        # Request with custom parameters
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 4096,
            "repetition_penalty": 1.2
        }

        make_llm_request(payload, "http://localhost:8000", None)

        # Verify create was called with client-specified params
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["extra_body"]["repetition_penalty"] == 1.2


def test_partial_client_params_do_not_pull_in_defaults(mock_llm_response):
    """A supplied provider parameter must not cause unrelated defaults to appear."""
    with patch('llm_router.llm_client._get_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_llm_response
        mock_get_client.return_value = mock_client

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        make_llm_request(payload, "http://localhost:8000", None)

        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["temperature"] == 0.7
        assert "top_p" not in call_kwargs
        assert "max_tokens" not in call_kwargs
        assert "extra_body" not in call_kwargs


def test_reasoning_effort_and_service_tier_are_sent_as_standard_chat_params(
    mock_llm_response,
):
    """Fast/plan controls should not be demoted into extra_body."""
    with patch('llm_router.llm_client._get_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_llm_response
        mock_get_client.return_value = mock_client

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "medium",
            "service_tier": "priority",
        }

        make_llm_request(payload, "http://localhost:8000", None)

        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["reasoning_effort"] == "medium"
        assert call_kwargs["service_tier"] == "priority"
        assert "extra_body" not in call_kwargs


def test_provider_error_status_and_body_are_preserved():
    """SDK status errors should remain diagnosable at the server boundary."""

    class ProviderStatusError(Exception):
        status_code = 400
        body = {
            "error": {
                "message": "bad request from provider",
                "type": "invalid_request_error",
                "code": "invalid_request_error",
            }
        }

    with patch("llm_router.llm_client._get_client") as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = ProviderStatusError()
        mock_get_client.return_value = mock_client

        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        with pytest.raises(LLMRequestError) as excinfo:
            make_llm_request(payload, "http://localhost:8000", None)

    error = excinfo.value
    assert error.status_code == 400
    assert error.message == "bad request from provider"
    assert error.body == ProviderStatusError.body


def test_env_vars_with_invalid_values():
    """Test that invalid env var values fall back to defaults."""
    from llm_router.llm_client import _get_env_float

    # Test that invalid values fall back to default
    with patch.dict(os.environ, {"TEST_INVALID": "not-a-number"}):
        result = _get_env_float("TEST_INVALID", 1.5)
        assert result == 1.5  # Should return the default

    # Test that valid values work
    with patch.dict(os.environ, {"TEST_VALID": "2.5"}):
        result = _get_env_float("TEST_VALID", 1.5)
        assert result == 2.5  # Should parse the env var
