"""Tests for default parameter handling in llm_client."""

import os
from unittest.mock import Mock, patch

from llm_router.llm_client import DEFAULT_PARAMS, make_llm_request


def test_default_params_applied_when_missing(mock_llm_response):
    """Test that default parameters are applied when client doesn't specify them."""
    with patch('llm_router.llm_client._get_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_llm_response
        mock_get_client.return_value = mock_client

        # Request without temperature, top_p, etc.
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}]
        }

        make_llm_request(payload, "http://localhost:8000", None)

        # Verify create was called with default params
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["temperature"] == DEFAULT_PARAMS["temperature"]
        assert call_kwargs["top_p"] == DEFAULT_PARAMS["top_p"]
        assert call_kwargs["max_tokens"] == DEFAULT_PARAMS["max_tokens"]
        assert call_kwargs["extra_body"]["repetition_penalty"] == DEFAULT_PARAMS["repetition_penalty"]


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


def test_partial_client_params_use_defaults(mock_llm_response):
    """Test that partial client params use defaults for missing ones."""
    with patch('llm_router.llm_client._get_client') as mock_get_client:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_llm_response
        mock_get_client.return_value = mock_client

        # Request with only some parameters
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,  # Only temperature specified
        }

        make_llm_request(payload, "http://localhost:8000", None)

        # Verify mixed usage
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        assert call_kwargs["temperature"] == 0.7  # Client value
        assert call_kwargs["top_p"] == DEFAULT_PARAMS["top_p"]  # Default
        assert call_kwargs["max_tokens"] == DEFAULT_PARAMS["max_tokens"]  # Default
        assert call_kwargs["extra_body"]["repetition_penalty"] == DEFAULT_PARAMS["repetition_penalty"]  # Default


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
