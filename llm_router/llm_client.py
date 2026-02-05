"""
LLM backend client module.

This module provides functions for making HTTP requests to the LLM backend.
"""

import os
from urllib.parse import urlparse

from openai import OpenAI

from .debug_log import log_debug


def get_request_timeout() -> float | None:
    """Get request timeout from environment variable.

    Returns:
        Timeout in seconds, or None if not set (no timeout)
    """
    timeout_str = os.environ.get("LLM_REQUEST_TIMEOUT")
    if not timeout_str:
        return None
    try:
        return float(timeout_str)
    except ValueError:
        return None


# Standard OpenAI chat completion parameters (whitelist)
OPENAI_PARAMS = {
    "temperature", "top_p", "n", "stream", "stop", "max_tokens",
    "max_completion_tokens", "presence_penalty", "frequency_penalty",
    "logit_bias", "user", "response_format", "seed", "tools", "tool_choice",
    "parallel_tool_calls", "logprobs", "top_logprobs",
}


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
    # Normalize base_url for OpenAI client compatibility
    # - If URL has no path (or just /), append /v1
    # - If URL already has a path (e.g., /v1 or /api/v3), keep as is
    base_url = llm_base_url.rstrip("/")
    parsed = urlparse(base_url)

    # Only append /v1 if path is empty
    if not parsed.path or parsed.path == "/":
        base_url = f"{base_url}/v1"

    client = OpenAI(
        base_url=base_url,
        api_key=api_key or "not-needed",
        timeout=get_request_timeout(),
    )

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
            "base_url": base_url,
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
