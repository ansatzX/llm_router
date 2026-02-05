"""
Debug logging module for LLM Router.

Provides centralized debug logging to file when DEBUG_MODE is enabled.
"""

import json
from datetime import datetime

DEBUG_MODE = False
DEBUG_LOG_FILE = "llm_router.log"


def set_debug_mode(enabled: bool):
    """Enable or disable debug mode."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


def log_debug(message: str, data: dict = None):
    """Log debug message to file if debug mode is enabled."""
    if not DEBUG_MODE:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] {message}\n")
        if data:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
            f.write("\n")
