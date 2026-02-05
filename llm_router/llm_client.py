"""
LLM backend client module.

This module provides functions for making HTTP requests to the LLM backend.
"""

import os
from openai import OpenAI


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


def make_llm_request(payload: dict, llm_base_url: str, api_key: str = None) -> dict:
    """
    Make a request to the LLM backend using OpenAI client.

    Args:
        payload: The request payload (OpenAI chat completions format)
        llm_base_url: The base URL of the LLM server
        api_key: Optional API key for authentication

    Returns:
        The response dict from the LLM backend

    Raises:
        Exception: If the request fails
    """
    client = OpenAI(
        base_url=f"{llm_base_url}/v1",
        api_key=api_key or "not-needed",
        timeout=get_request_timeout(),
    )

    try:
        # Copy payload to avoid modifying original
        params = payload.copy()
        model = params.pop("model", "default")
        messages = params.pop("messages", [])

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **params  # Pass all other parameters directly
        )

        # Convert to dict format
        return response.model_dump()
    except Exception as e:
        raise Exception(f"LLM request error: {e}")
