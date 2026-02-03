"""
Model configuration and content interception module.

This module provides functionality for:
1. Loading model configuration from environment variables
2. Detecting model type (text vs multimodal)
3. Intercepting and validating uploaded content based on model capabilities
"""

import os
import base64
import mimetypes
from typing import Optional, Tuple, Dict, Any, Literal


# Valid model types
VALID_MODEL_TYPES = ("text", "multimodal")


def validate_model_type(value: str) -> Literal["text", "multimodal"]:
    """Validate and normalize model type value.

    Args:
        value: The model type string to validate

    Returns:
        Normalized model type ("text" or "multimodal")
    """
    normalized = value.lower().strip()
    return normalized if normalized in VALID_MODEL_TYPES else "text"


def get_model_type() -> Literal["text", "multimodal"]:
    """
    Get the model type from environment variable.

    Returns:
        "text" or "multimodal"
    """
    return validate_model_type(os.environ.get("MODEL_TYPE", "text"))


def is_multimodal() -> bool:
    """Check if the configured model is multimodal."""
    return get_model_type() == "multimodal"


def parse_int_env(env_var: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    """Parse integer environment variable with validation.

    Args:
        env_var: Name of the environment variable
        default: Default value if not set or invalid
        min_value: Optional minimum allowed value
        max_value: Optional maximum allowed value

    Returns:
        Parsed and validated integer value
    """
    try:
        value = int(os.environ.get(env_var, str(default)))
        if min_value is not None:
            value = max(value, min_value)
        if max_value is not None:
            value = min(value, max_value)
        return value
    except (ValueError, TypeError):
        return default


def get_max_upload_size() -> int:
    """
    Get maximum upload size in bytes.

    Returns:
        Maximum size in bytes (default: 10MB)
    """
    max_mb = parse_int_env("MAX_UPLOAD_SIZE_MB", default=10, min_value=1, max_value=100)
    return max_mb * 1024 * 1024




def get_text_model_media_prompt() -> str:
    """Get the prompt returned when text model receives media."""
    return os.environ.get(
        "TEXT_MODEL_MEDIA_PROMPT",
        "This model cannot process images or documents. Please use a local tool (e.g., MCP tool) to analyze the content and pass the results as text."
    )


def get_multimodal_document_prompt() -> str:
    """Get the prompt returned when multimodal model receives document."""
    return os.environ.get(
        "MULTIMODAL_MODEL_DOCUMENT_PROMPT",
        "This model cannot process documents directly. Please use a local tool (e.g., MCP tool) to extract the content and pass it as text."
    )


def detect_content_type(content_block: dict) -> Tuple[str, Optional[str]]:
    """
    Detect the type of content in a message content block.

    Args:
        content_block: A content block from Anthropic or OpenAI format

    Returns:
        Tuple of (content_type, file_format)
        - content_type: "text", "image", "document", or "unknown"
        - file_format: File extension if detected (e.g., "png", "pdf")
    """
    block_type = content_block.get("type", "").lower()

    if block_type == "text":
        return ("text", None)

    elif block_type == "image":
        # Anthropic image format
        source = content_block.get("source", {})
        if source.get("type") == "base64":
            media_type = source.get("media_type", "")
            if "png" in media_type:
                return ("image", "png")
            elif "jpeg" in media_type or "jpg" in media_type:
                return ("image", "jpg")
            elif "webp" in media_type:
                return ("image", "webp")
            elif "gif" in media_type:
                return ("image", "gif")
        return ("image", "unknown")

    elif block_type == "image_url":
        # OpenAI image format
        url = content_block.get("url", "")
        if url.startswith("data:image/"):
            # Data URL format
            mime = url.split(";")[0].replace("data:image/", "")
            return ("image", mime)
        return ("image", "url")

    elif block_type in ("file", "document"):
        # Document content
        source = content_block.get("source", {})
        if source.get("type") == "base64":
            mime_type = source.get("mime_type", "")
            if "pdf" in mime_type:
                return ("document", "pdf")
            elif "text" in mime_type:
                return ("document", "txt")
            elif "word" in mime_type or "doc" in mime_type:
                return ("document", "doc")
        return ("document", "unknown")

    return ("unknown", None)


def _check_image_content(model_type: str) -> Tuple[bool, Optional[dict]]:
    """Check if image content is allowed for the model type."""
    if model_type == "text":
        return (False, {
            "type": "text_model_rejection",
            "message": get_text_model_media_prompt()
        })
    return (True, None)


def _check_document_content(model_type: str) -> Tuple[bool, Optional[dict]]:
    """Check if document content is allowed for the model type."""
    if model_type == "multimodal":
        return (False, {
            "type": "multimodal_document_rejection",
            "message": get_multimodal_document_prompt()
        })
    return (False, {
        "type": "text_model_rejection",
        "message": get_text_model_media_prompt()
    })


def validate_content_blocks(
    content_blocks: list,
    is_anthropic_format: bool = False
) -> Tuple[bool, Optional[dict]]:
    """
    Validate a list of content blocks based on model capabilities.

    Args:
        content_blocks: List of content blocks (Anthropic or OpenAI format)
        is_anthropic_format: True if Anthropic format, False if OpenAI

    Returns:
        Tuple of (is_valid, response_data)
        - is_valid: True if all content can be processed
        - response_data: If not valid, contains rejection response
    """
    model_type = get_model_type()

    content_validators = {
        "text": lambda: (True, None),
        "image": lambda: _check_image_content(model_type),
        "document": lambda: _check_document_content(model_type),
    }

    for block in content_blocks:
        content_type, _ = detect_content_type(block)
        validator = content_validators.get(content_type, lambda: (True, None))
        is_valid, response_data = validator()
        if not is_valid:
            return (False, response_data)

    return (True, None)
