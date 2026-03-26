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


def log_debug(message: str, data: dict = None):
    """Log debug message to file if debug mode is enabled."""
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
        _LOG_FILE_HANDLE.write(json.dumps(data, ensure_ascii=False, indent=2))
        _LOG_FILE_HANDLE.write("\n")
    _LOG_FILE_HANDLE.flush()  # Ensure immediate write
