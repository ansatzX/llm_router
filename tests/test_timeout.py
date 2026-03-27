"""Tests for timeout configuration in llm_client module."""

import os
from unittest.mock import patch

import httpx

from llm_router.llm_client import get_request_timeout


class TestGetRequestTimeout:
    """Test timeout configuration from environment variables."""

    def test_default_timeout_values(self, reset_llm_client_state):
        """Test default timeout configuration when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove all timeout env vars
            for key in ['LLM_CONNECT_TIMEOUT', 'LLM_READ_TIMEOUT',
                       'LLM_WRITE_TIMEOUT', 'LLM_POOL_TIMEOUT']:
                os.environ.pop(key, None)

            timeout = get_request_timeout()

            # Should return httpx.Timeout with default values
            assert isinstance(timeout, httpx.Timeout)
            assert timeout.connect == 5.0
            assert timeout.read == 120.0
            assert timeout.write == 30.0
            assert timeout.pool == 10.0

    def test_custom_timeout_from_env(self, reset_llm_client_state):
        """Test custom timeout from environment variables."""
        with patch.dict(os.environ, {
            'LLM_CONNECT_TIMEOUT': '10',
            'LLM_READ_TIMEOUT': '300',
            'LLM_WRITE_TIMEOUT': '60',
            'LLM_POOL_TIMEOUT': '20'
        }):
            timeout = get_request_timeout()

            # Should return custom values
            assert timeout.connect == 10.0
            assert timeout.read == 300.0
            assert timeout.write == 60.0
            assert timeout.pool == 20.0

    def test_invalid_timeout_falls_back_to_default(self, reset_llm_client_state):
        """Test that invalid timeout value falls back to default."""
        with patch.dict(os.environ, {
            'LLM_READ_TIMEOUT': 'invalid',
            'LLM_CONNECT_TIMEOUT': 'also_invalid'
        }):
            timeout = get_request_timeout()

            # Should return default values
            assert timeout.read == 120.0
            assert timeout.connect == 5.0

    def test_timeout_is_cached(self, reset_llm_client_state):
        """Test that timeout is parsed only once and cached."""
        with patch.dict(os.environ, {'LLM_READ_TIMEOUT': '200'}):
            # First call
            timeout1 = get_request_timeout()
            assert timeout1.read == 200.0

        # Second call should return cached value (even though env var changed)
        with patch.dict(os.environ, {'LLM_READ_TIMEOUT': '300'}):
            timeout2 = get_request_timeout()
            # Should still be 200.0 (cached)
            assert timeout2.read == 200.0

    def test_float_timeout_value(self, reset_llm_client_state):
        """Test that float timeout values are properly parsed."""
        with patch.dict(os.environ, {
            'LLM_CONNECT_TIMEOUT': '5.5',
            'LLM_READ_TIMEOUT': '123.45'
        }):
            timeout = get_request_timeout()

            # Should parse float correctly
            assert timeout.connect == 5.5
            assert timeout.read == 123.45

    def test_zero_timeout_is_valid(self, reset_llm_client_state):
        """Test that zero timeout (immediate timeout) is a valid value."""
        with patch.dict(os.environ, {'LLM_READ_TIMEOUT': '0'}):
            timeout = get_request_timeout()

            # Zero is valid (though not recommended)
            assert timeout.read == 0.0

    def test_timeout_object_type(self, reset_llm_client_state):
        """Test that timeout returns correct httpx.Timeout object."""
        timeout = get_request_timeout()

        # Should be httpx.Timeout instance
        assert isinstance(timeout, httpx.Timeout)
        assert hasattr(timeout, 'connect')
        assert hasattr(timeout, 'read')
        assert hasattr(timeout, 'write')
        assert hasattr(timeout, 'pool')
