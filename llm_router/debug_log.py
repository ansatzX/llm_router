"""
Debug logging module for LLM Router.

Provides centralized debug logging to file when DEBUG_MODE is enabled.
"""

import json
from datetime import datetime

DEBUG_MODE = False
DEBUG_LOG_FILE = "llm_router.log"

# Keep file handle open during debug mode for better I/O performance
_LOG_FILE_HANDLE = None


def set_debug_mode(enabled: bool):
    """Enable or disable debug mode."""
    global DEBUG_MODE, _LOG_FILE_HANDLE
    DEBUG_MODE = enabled

    # Close any existing file handle
    if _LOG_FILE_HANDLE:
        _LOG_FILE_HANDLE.close()
        _LOG_FILE_HANDLE = None

    # Open file handle if debug mode is enabled
    if enabled:
        _LOG_FILE_HANDLE = open(DEBUG_LOG_FILE, "a", encoding="utf-8")


def _truncate_large_content(data: dict, max_length: int = 500) -> dict:
    """Truncate large content fields for readability."""
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_length:
            result[key] = value[:max_length] + f"... (truncated, {len(value)} total chars)"
        elif isinstance(value, list):
            result[key] = [_truncate_large_content(item, max_length) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            result[key] = _truncate_large_content(value, max_length)
        else:
            result[key] = value
    return result


def log_debug(message: str, data: dict = None, truncate: bool = True):
    """Log debug message to file if debug mode is enabled.

    Args:
        message: Log message/label
        data: Optional data dictionary to log
        truncate: If True, truncate large content fields for readability
    """
    if not DEBUG_MODE:
        return

    global _LOG_FILE_HANDLE

    # Re-open file if handle was closed
    if _LOG_FILE_HANDLE is None:
        _LOG_FILE_HANDLE = open(DEBUG_LOG_FILE, "a", encoding="utf-8")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _LOG_FILE_HANDLE.write(f"\n{'='*80}\n")
    _LOG_FILE_HANDLE.write(f"[{timestamp}] {message}\n")

    if data:
        # Truncate large content for readability
        log_data = _truncate_large_content(data) if truncate else data

        # Add summary for messages
        if "messages" in log_data:
            messages = log_data.get("messages", [])
            _LOG_FILE_HANDLE.write(f"Message Count: {len(messages)}\n")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, str):
                    preview = content[:100] + "..." if len(content) > 100 else content
                    _LOG_FILE_HANDLE.write(f"  [{i}] {role}: {preview}\n")

        # Add summary for choices
        if "choices" in log_data:
            choices = log_data.get("choices", [])
            _LOG_FILE_HANDLE.write(f"Choices Count: {len(choices)}\n")

        # Write full JSON
        _LOG_FILE_HANDLE.write("\nFull Data:\n")
        _LOG_FILE_HANDLE.write(json.dumps(log_data, ensure_ascii=False, indent=2))
        _LOG_FILE_HANDLE.write("\n")

    _LOG_FILE_HANDLE.flush()  # Ensure immediate write
