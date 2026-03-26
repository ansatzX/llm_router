"""
LLM backend client module.

This module provides functions for making HTTP requests to the LLM backend.
"""

import os
from urllib.parse import urlparse

from openai import OpenAI

from .debug_log import log_debug


# Cached timeout value (parsed once at startup)
_REQUEST_TIMEOUT: float | None = None
_TIMEOUT_PARSED = False  # Flag to track if we've attempted parsing


def get_request_timeout() -> float | None:
    """Get request timeout from environment variable.

    Returns:
        Timeout in seconds, or None if not set or invalid
    """
    global _REQUEST_TIMEOUT, _TIMEOUT_PARSED

    if not _TIMEOUT_PARSED:
        _TIMEOUT_PARSED = True
        timeout_str = os.environ.get("LLM_REQUEST_TIMEOUT")
        if timeout_str:
            try:
                _REQUEST_TIMEOUT = float(timeout_str)
            except ValueError:
                # Invalid value, keep as None
                pass

    return _REQUEST_TIMEOUT


# Standard OpenAI chat completion parameters (whitelist)
OPENAI_PARAMS = {
    "temperature", "top_p", "n", "stream", "stop", "max_tokens",
    "max_completion_tokens", "presence_penalty", "frequency_penalty",
    "logit_bias", "user", "response_format", "seed", "tools", "tool_choice",
    "parallel_tool_calls", "logprobs", "top_logprobs",
}


# Default API key when none provided
_DEFAULT_API_KEY = "not-needed"

# Cached OpenAI client instance (keyed by base_url and api_key)
# Note: This cache is unbounded but acceptable for single-backend usage.
# If multi-backend support is needed, consider adding LRU eviction with max size.
_CLIENT_CACHE: dict[tuple[str, str], OpenAI] = {}


def _get_client(llm_base_url: str, api_key: str | None) -> OpenAI:
    """Get or create cached OpenAI client instance.

    Args:
        llm_base_url: The base URL of the LLM server
        api_key: Optional API key for authentication

    Returns:
        OpenAI client instance
    """
    # Normalize base_url for OpenAI client compatibility
    base_url = llm_base_url.rstrip("/")
    parsed = urlparse(base_url)

    # Only append /v1 if path is empty
    if not parsed.path or parsed.path == "/":
        base_url = f"{base_url}/v1"

    # Use tuple of (base_url, api_key) as cache key
    cache_key = (base_url, api_key or _DEFAULT_API_KEY)

    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = OpenAI(
            base_url=base_url,
            api_key=api_key or _DEFAULT_API_KEY,
            timeout=get_request_timeout(),
        )

    return _CLIENT_CACHE[cache_key]


def make_llm_request(payload: dict, llm_base_url: str, api_key: str = None) -> dict:
    """
    Make a request to the LLM backend using OpenAI client.

    Args:
        payload: The request payload (OpenAI chat completions format)
        llm_base_url: The base URL of the LLM server (e.g., http://host:port or http://host:port/v1)
        api_key: Optional API key for authentication

    Returns:
        The response dict from the LLM backend

    Raises:
        Exception: If the request fails
    """
    client = _get_client(llm_base_url, api_key)

    try:
        # Copy payload to avoid modifying original
        params = payload.copy()
        model = params.pop("model", "default")
        messages = params.pop("messages", [])

        # Separate standard OpenAI params from extra params (vLLM/SGLang specific)
        openai_params = {k: v for k, v in params.items() if k in OPENAI_PARAMS}
        extra_params = {k: v for k, v in params.items() if k not in OPENAI_PARAMS}

        # Pass extra params via extra_body for vLLM/SGLang backends
        if extra_params:
            openai_params["extra_body"] = extra_params

        # Log request to LLM backend
        log_debug("LLM_REQUEST", {
            "base_url": client.base_url,
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
        raise Exception(f"LLM request error: {e}")


def list_models(llm_base_url: str, api_key: str = None) -> dict:
    """
    List available models from the LLM backend.

    Args:
        llm_base_url: The base URL of the LLM server
        api_key: Optional API key for authentication

    Returns:
        Models list response from the LLM backend
    """
    client = _get_client(llm_base_url, api_key)

    try:
        response = client.models.list()
        return response.model_dump()
    except Exception as e:
        # Return default model list on error (backend may not support /models endpoint)
        return {
            "object": "list",
            "data": [{"id": "default", "object": "model", "created": 0}]
        }
