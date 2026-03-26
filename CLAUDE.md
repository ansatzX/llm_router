# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Router is a Flask-based API proxy that enables standard OpenAI API clients to work with LLM backends using MCP XML format for tool calls (e.g., MiroThinker on SGLang). It handles protocol translation and tool format conversion.

## Development Commands

### Installation
```bash
# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Using uv (recommended)
uv venv
uv pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### Running the Server
```bash
# Configure via .env file (copy from .env.example)
cp .env.example .env
# Edit .env with your settings

llm-router                    # Run on default port (5001)
llm-router --port 8080        # Custom port
llm-router --debug            # Enable debug logging to llm_router.log
```

### Testing
```bash
pytest                        # Run tests
pytest -v                     # Verbose output
uv run pytest -v              # Using uv
```

### Linting
```bash
ruff check .                  # Run linter
ruff check --fix .            # Auto-fix issues
```

## Architecture

The router acts as a protocol translator between OpenAI clients and backend LLMs:

```
Client (OpenAI format)
    ↓ Tool definitions → MCP System Prompt
    ↓
Backend LLM responds with tool calls (MCP XML or JSON format)
    ↓ Router parses and converts
    ↓
Client receives standard OpenAI API response
```

### Core Modules

**`server.py`** - Flask endpoints and request orchestration
- Handles `/v1/chat/completions` (OpenAI format)
- Injects MCP system prompt when tools are provided
- Forces `stream=False` when tools present (needs full response for parsing)

**`mcp_converter.py`** - Format conversion utilities
- Converts OpenAI tools → MCP system prompt
- Parses 3 tool call formats: MCP XML (`<use_mcp_tool>`), Tool call XML (`[TOOL_CALL]`), JSON
- Converts tool calls back to OpenAI format

**`llm_client.py`** - Backend communication
- Uses OpenAI SDK to communicate with backend
- Separates standard OpenAI params from non-standard params (e.g., `repetition_penalty`, `top_k`)
- Passes non-standard params via `extra_body` for vLLM/SGLang compatibility

**`cli.py`** - Entry point
- Loads `.env` file via python-dotenv
- Starts Flask server

## Key Concepts

### Tool Call Format Support

The router parses multiple formats from backend LLMs:

1. **MCP XML**: `<use_mcp_tool><server_name>...</server_name><tool_name>...</tool_name><arguments>...</arguments></use_mcp_tool>`
2. **Tool call XML**: `[TOOL_CALL]<tool_name>...</tool_name><arguments>...</arguments>[/TOOL_CALL]`
3. **JSON**: `{"name": "...", "arguments": {...}}`

All converted to standard OpenAI tool call format.

### vLLM/SGLang Parameters

Non-standard parameters are passed via `extra_body`:
- Standard params: `temperature`, `top_p`, `max_tokens`, etc.
- Non-standard params: `repetition_penalty`, `top_k`, `min_p`, etc.
- Streaming is always disabled internally when tools present (`stream=False`)

## Configuration

Key environment variables (set in `.env`):

- `LLM_BASE_URL` - Backend LLM URL (default: `http://localhost:8000`)
- `LLM_API_KEY` - Backend API key (optional)
- `FLASK_PORT` - Router port (default: 5001)

## Debugging

Run with `--debug` flag to create `llm_router.log` with:
- `CLIENT_REQUEST` - Original request from client
- `LLM_REQUEST` - Request sent to backend (after MCP prompt injection)
- `LLM_RESPONSE` - Raw response from backend
- `CLIENT_RESPONSE` - Final response sent to client

## Code Style

- Python 3.10+ with type hints
- Line length: 100 characters
- Linter: Ruff with rules E, F, I, N, W, UP, B, C4, SIM
- Docstring convention: Google style
