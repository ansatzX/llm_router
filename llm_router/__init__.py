"""
LLM Router - A Flask-based router for LLM APIs with OpenAI and Anthropic protocol support.

This package provides a Flask application that acts as a proxy between LLM backends
and clients expecting standard OpenAI or Anthropic protocol formats.

Example:
    >>> from llm_router import create_app
    >>> app = create_app()
    >>> app.run(host="0.0.0.0", port=5001)

Attributes:
    __version__: The version string of the package.
    __author__: The author of the package.
    __license__: The license of the package.
"""

__version__ = "0.1.0"
__author__ = "ansatz Vibe"
__license__ = "MIT"
__all__ = ["create_app", "app", "set_debug_mode"]

from .server import create_app, app, set_debug_mode
