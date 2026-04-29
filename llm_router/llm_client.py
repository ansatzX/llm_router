"""LLM backend client module.

This module provides functions for making HTTP requests to the LLM backend
using the OpenAI SDK with support for non-standard parameters.
"""

import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import OpenAI

from llm_router.debug_log import log_debug

# Thread-safe initialization lock
_INIT_LOCK = threading.Lock()

# Cached timeout value (parsed once at startup)
_REQUEST_TIMEOUT: httpx.Timeout | None = None


def get_request_timeout() -> httpx.Timeout:
    """Get request timeout configuration.

    Returns:
        httpx.Timeout with separate connect, read, write, and pool timeouts.
        Defaults: connect=5s, read=120s, write=30s, pool=10s

    Note:
        This function is thread-safe and uses double-checked locking
        to ensure timeout is initialized exactly once.
    """
    global _REQUEST_TIMEOUT

    # Double-checked locking pattern for thread safety
    if _REQUEST_TIMEOUT is None:
        with _INIT_LOCK:
            # Check again inside lock
            if _REQUEST_TIMEOUT is None:
                # Fine-grained timeout configuration
                connect_timeout = _get_env_float("LLM_CONNECT_TIMEOUT", 5.0)
                read_timeout = _get_env_float("LLM_READ_TIMEOUT", 120.0)
                write_timeout = _get_env_float("LLM_WRITE_TIMEOUT", 30.0)
                pool_timeout = _get_env_float("LLM_POOL_TIMEOUT", 10.0)

                _REQUEST_TIMEOUT = httpx.Timeout(
                    connect=connect_timeout,  # Time to establish connection
                    read=read_timeout,        # Time to wait for data (no activity)
                    write=write_timeout,      # Time to write request
                    pool=pool_timeout         # Time to get connection from pool
                )

    return _REQUEST_TIMEOUT


def _get_env_float(key: str, default: float) -> float:
    """Safely parse float from environment variable.

    Args:
        key: Environment variable name.
        default: Default float value if variable is not set or invalid.

    Returns:
        Parsed float value from environment or default.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(key: str, default: int) -> int:
    """Safely parse int from environment variable.

    Args:
        key: Environment variable name.
        default: Default int value if variable is not set or invalid.

    Returns:
        Parsed int value from environment or default.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Standard OpenAI chat completion parameters (whitelist)
OPENAI_PARAMS: set[str] = {
    "temperature", "top_p", "n", "stream", "stop", "max_tokens",
    "max_completion_tokens", "presence_penalty", "frequency_penalty",
    "logit_bias", "user", "response_format", "seed", "tools", "tool_choice",
    "parallel_tool_calls", "logprobs", "top_logprobs",
}


# Default parameter values - MiroThinker recommendations for optimal tool-calling performance
DEFAULT_PARAMS: dict[str, float | int] = {
    "temperature": _get_env_float("DEFAULT_TEMPERATURE", 1.0),
    "top_p": _get_env_float("DEFAULT_TOP_P", 0.95),
    "max_tokens": _get_env_int("DEFAULT_MAX_TOKENS", 16384),
    "repetition_penalty": _get_env_float("DEFAULT_REPETITION_PENALTY", 1.05),
}


# Default API key when none provided
_DEFAULT_API_KEY: str = "not-needed"

# Cached OpenAI client instance (keyed by base_url and api_key)
# Note: This cache is unbounded but acceptable for single-backend usage.
# If multi-backend support is needed, consider adding LRU eviction with max size.
_CLIENT_CACHE: dict[tuple[str, str], OpenAI] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def _get_client(llm_base_url: str, api_key: str | None) -> OpenAI:
    """Get or create cached OpenAI client instance.

    Args:
        llm_base_url: The base URL of the LLM server.
        api_key: Optional API key for authentication.

    Returns:
        OpenAI client instance configured for the backend.

    Note:
        This function is thread-safe and uses double-checked locking
        to ensure client is created exactly once per (base_url, api_key) pair.
    """
    # Use original URL as cache key (no parsing needed for cache hit)
    cache_key = (llm_base_url, api_key or _DEFAULT_API_KEY)

    # Double-checked locking pattern for thread safety
    if cache_key not in _CLIENT_CACHE:
        with _CLIENT_CACHE_LOCK:
            # Check again inside lock
            if cache_key not in _CLIENT_CACHE:
                # Normalize base_url only on cache miss
                base_url = llm_base_url.rstrip("/")
                parsed = urlparse(base_url)

                # Only append /v1 if path is empty
                if not parsed.path or parsed.path == "/":
                    base_url = f"{base_url}/v1"

                _CLIENT_CACHE[cache_key] = OpenAI(
                    base_url=base_url,
                    api_key=api_key or _DEFAULT_API_KEY,
                    timeout=get_request_timeout(),
                )

    return _CLIENT_CACHE[cache_key]


def make_llm_request(payload: dict[str, Any], llm_base_url: str, api_key: str | None = None) -> dict[str, Any]:
    """Make a request to the LLM backend using OpenAI client.

    Args:
        payload: The request payload in OpenAI chat completions format.
        llm_base_url: The base URL of the LLM server (e.g., http://host:port or http://host:port/v1).
        api_key: Optional API key for authentication.

    Returns:
        The response dict from the LLM backend.

    Raises:
        Exception: If the request fails.
    """
    client = _get_client(llm_base_url, api_key)

    try:
        # Apply defaults only for missing parameters (efficient merge)
        params = payload.copy()
        for key, value in DEFAULT_PARAMS.items():
            if key not in params:
                params[key] = value

        model = params.pop("model", "default")
        messages = params.pop("messages", [])

        # Separate standard OpenAI params from extra params for vLLM/SGLang
        openai_params = {}
        extra_params = {}
        for k, v in params.items():
            if k in OPENAI_PARAMS:
                openai_params[k] = v
            else:
                extra_params[k] = v

        if extra_params:
            openai_params["extra_body"] = extra_params

        # Log request to LLM backend
        log_debug("LLM_REQUEST", {
            "base_url": str(client.base_url),
            "model": model,
            "messages": messages,
            "openai_params": {k: v for k, v in openai_params.items() if k != "extra_body"},
            "extra_params": extra_params if extra_params else None,
        })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **openai_params
        )

        # Convert to dict format
        result = response.model_dump()

        # Log response from LLM backend
        log_debug("LLM_RESPONSE", result)

        return result
    except Exception as e:
        raise Exception(f"LLM request error: {e}") from e


def list_models(llm_base_url: str, api_key: str | None = None) -> dict[str, Any]:
    """List available models from the LLM backend.

    Args:
        llm_base_url: The base URL of the LLM server.
        api_key: Optional API key for authentication.

    Returns:
        Models list response from the LLM backend with default fallback.
    """
    client = _get_client(llm_base_url, api_key)

    try:
        response = client.models.list()
        return response.model_dump()
    except Exception:
        # Return default model list on error (backend may not support /models endpoint)
        return {
            "object": "list",
            "data": [{"id": "default", "object": "model", "created": 0}]
        }


def check_backend_health(
    llm_base_url: str,
    timeout: float | None = None,
    api_key: str | None = None
) -> dict[str, Any]:
    """Check if backend LLM server is healthy.

    Makes HTTP request to backend health endpoints with fallback chain.
    Used for deployment platform health checks and readiness probes.

    Fallback order (most comprehensive to least):
    1. /health_generate - Generates token (most reliable)
    2. /readiness - Kubernetes readiness probe
    3. /health - Basic health check
    4. /metrics - Prometheus metrics

    Args:
        llm_base_url: Base URL of LLM backend.
        timeout: Health check timeout in seconds (default: from HEALTH_CHECK_TIMEOUT env).
        api_key: Optional API key (unused for health checks).

    Returns:
        Dict with keys:
        - healthy: bool - Whether backend is responding
        - latency_ms: float | None - Response latency in milliseconds
        - error: str | None - Error message if unhealthy
        - endpoint: str - Which endpoint was used
    """
    # Get timeout from environment or use default
    if timeout is None:
        timeout = _get_env_float("HEALTH_CHECK_TIMEOUT", 3.0)

    # Normalize base URL
    base_url = llm_base_url.rstrip("/")

    # Try endpoints in order of comprehensiveness
    # /health_generate is most reliable (actually generates a token)
    # /readiness checks worker availability
    # /health is basic check
    # /metrics as fallback (always available in SGLang)
    endpoints_to_try = [
        "/health_generate",  # Most comprehensive - verifies token generation
        "/readiness",         # Kubernetes readiness - checks workers
        "/health",            # Basic health check
        "/metrics"            # Prometheus metrics (fallback, always available)
    ]

    for endpoint in endpoints_to_try:
        url = f"{base_url}{endpoint}"

        try:
            start_time = time.time()

            with httpx.Client(timeout=timeout) as client:
                response = client.get(url)

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {
                    "healthy": True,
                    "latency_ms": round(latency_ms, 2),
                    "error": None,
                    "endpoint": endpoint
                }
            elif response.status_code == 404:
                # Try next endpoint
                continue
            else:
                # HTTP error
                return {
                    "healthy": False,
                    "latency_ms": round(latency_ms, 2),
                    "error": f"Backend returned HTTP {response.status_code}",
                    "endpoint": endpoint
                }

        except httpx.TimeoutException:
            # Don't retry on timeout - try next endpoint
            continue
        except httpx.ConnectError:
            # Connection refused - backend not up yet
            return {
                "healthy": False,
                "latency_ms": None,
                "error": "Connection refused",
                "endpoint": endpoint
            }
        except Exception:
            # Try next endpoint on other errors
            continue

    # All endpoints failed
    return {
        "healthy": False,
        "latency_ms": None,
        "error": "All health check endpoints failed",
        "endpoint": "none"
    }
