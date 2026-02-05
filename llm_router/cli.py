#!/usr/bin/env python3
"""CLI entry point that loads .env before importing the server module."""

import os
import sys


def main():
    """Entry point for the CLI command."""
    # Load .env FIRST, before any other imports from llm_router
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    # Parse args before importing server (to avoid loading server for --help)
    import argparse
    parser = argparse.ArgumentParser(description="LLM Router - API proxy for LLM backends")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to llm_router.log")
    parser.add_argument("--port", type=int, default=None, help="Port to run on (default: 5001)")
    args = parser.parse_args()

    # NOW import server (with .env already loaded)
    from llm_router.server import app, set_debug_mode

    if args.debug:
        set_debug_mode(True)
        print("[DEBUG] Debug mode enabled, logging to llm_router.log")

    port = args.port or int(os.environ.get("FLASK_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
