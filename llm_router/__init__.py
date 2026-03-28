"""LLM Router - A Flask-based router for LLM APIs with OpenAI protocol support.

This package provides a Flask application that acts as a proxy between LLM backends
and clients expecting standard OpenAI protocol formats with MCP XML tool parsing.

Example:
    >>> from llm_router import create_app
    >>> app = create_app()
    >>> app.run(host='0.0.0.0', port=5001)

Attributes:
    __version__ (str): The version string of the package.
    __author__ (str): The author of the package.
    __license__ (str): The license of the package.
"""

__version__: str = "0.1.0"
__author__: str = "ansatz Vibe"
__license__: str = "MIT"
__all__: list[str] = ["create_app", "app", "set_debug_mode"]

from llm_router.debug_log import set_debug_mode
from llm_router.server import app, create_app
