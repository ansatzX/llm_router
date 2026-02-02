"""
LLM backend client module.

This module provides functions for making HTTP requests to the LLM backend.
"""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def make_llm_request(payload: dict, llm_base_url: str, api_key: str = None) -> dict:
    """
    Make an HTTP request to the LLM backend.

    Args:
        payload: The request payload to send to the LLM backend
        llm_base_url: The base URL of the LLM server
        api_key: Optional API key for authentication

    Returns:
        The JSON response from the LLM backend

    Raises:
        Exception: If the HTTP request fails
    """
    url = f"{llm_base_url}/v1/chat/completions"
    data = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
    }

    # Add API key if provided
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    req = Request(url, data=data, headers=headers)

    try:
        with urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        error_data = e.read().decode('utf-8')
        raise Exception(f"LLM HTTP error {e.code}: {error_data}")
    except URLError as e:
        raise Exception(f"LLM connection error: {e.reason}")
