# LLM Router

A Flask-based API proxy that enables standard OpenAI/Anthropic API clients to work with LLM backends using MCP XML format for tool calls (e.g., MiroThinker on SGLang).

## Features

- **Protocol Translation**: Accepts OpenAI and Anthropic API formats
- **MCP Tool Conversion**: Converts API tools to MCP system prompt, parses responses back to standard formats
- **Dual Format Support**: Parses both MCP XML (`<use_mcp_tool>`) and JSON (`{"name": "...", "arguments": {...}}`) tool calls
- **Lazy Tool Loading**: When tools exceed context limit, uses progressive loading via `search_tools`
- **Think Tag Handling**: Strips `<think>` reasoning tags from model output
- **Content Validation**: Intercepts media content based on model capabilities (text vs multimodal)
- **JSON Repair**: Handles malformed JSON in tool arguments
- **vLLM/SGLang Support**: Non-standard parameters (e.g., `repetition_penalty`, `top_k`) passed via `extra_body`
- **Non-Streaming**: Streaming disabled internally (router requires full response for tool parsing)

## Installation

```bash
# Using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Using uv (recommended)
uv venv
uv pip install -e .
```

## Usage

```bash
# Configure via .env file (recommended)
cp .env.example .env
# Edit .env with your settings
llm-router

# Or via environment variables
export LLM_BASE_URL="http://your-llm-backend:8000"
llm-router

# With command line options
llm-router --port 8080

# Enable debug logging to llm_router.log
llm-router --debug
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI Chat Completions |
| `/v1/messages` | POST | Anthropic Messages |
| `/v1/chat` | POST | Unified (auto-detects format) |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## How It Works

### Basic Flow

```
Client (OpenAI/Anthropic format)
        ↓
    Tool definitions → MCP System Prompt
        ↓
    Backend LLM responds with tool calls
    (MCP XML or JSON format)
        ↓
    Router parses and converts
        ↓
Client receives standard API response
```

### Tool Call Format Support

**MCP XML format**:
```xml
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>{"location": "London"}</arguments>
</use_mcp_tool>
```

**Tool call XML format**:
```xml
<tool_call>
<tool_name>get_weather</tool_name>
<arguments>{"location": "London"}</arguments>
</tool_call>
```

**Simple XML format**:
```xml
<tool>
<function_name>get_weather</function_name>
<arguments>{"location": "London"}</arguments>
</tool>
```

**JSON format**:
```json
{"name": "get_weather", "arguments": {"location": "London"}}
```

All formats are converted to standard API responses:

**OpenAI format:**
```json
{
  "tool_calls": [{
    "id": "call_xxx",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"London\"}"
    }
  }]
}
```

**Anthropic format:**
```json
{
  "content": [{
    "type": "tool_use",
    "id": "toolu_xxx",
    "name": "get_weather",
    "input": {"location": "London"}
  }]
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:8000` | Backend LLM URL |
| `LLM_API_KEY` | - | Backend API key (optional) |
| `FLASK_PORT` | `5001` | Router port |
| `MODEL_TYPE` | `text` | `text` or `multimodal` |
| `MAX_TOKENS_CAP` | `16384` | Max completion tokens cap |
| `MAX_TOOLS_CHARS` | `40000` | Threshold for lazy tool loading |
| `MAX_TOOL_SEARCH_ROUNDS` | `20` | Max rounds for search_tools loop |
| `MAX_CONTEXT_CHARS` | `800000` | Max message content chars (~200k tokens) |
| `LLM_REQUEST_TIMEOUT` | - | Request timeout in seconds |

## vLLM/SGLang Parameters

The router automatically handles non-standard parameters used by vLLM/SGLang backends:

- **Standard OpenAI parameters** (e.g., `temperature`, `top_p`, `max_tokens`) are passed directly
- **Non-standard parameters** (e.g., `repetition_penalty`, `top_k`, `min_p`) are passed via `extra_body`
- **Streaming is disabled** internally - the router needs full responses for MCP tool parsing

Example request with vLLM parameters:
```json
{
  "model": "model-name",
  "messages": [...],
  "temperature": 0.7,
  "repetition_penalty": 1.05,
  "top_k": 50
}
```

The router will send to backend:
```python
client.chat.completions.create(
    model="model-name",
    messages=[...],
    temperature=0.7,
    stream=False,  # forced
    extra_body={"repetition_penalty": 1.05, "top_k": 50}
)
```

## Message Truncation

When conversation history exceeds `MAX_CONTEXT_CHARS`, the router automatically truncates messages:

- **Preserves first message** (system prompt)
- **Preserves last message** (current request)
- **Removes older messages** from the middle
- Keeps most recent context for continuity

This prevents context overflow errors when:
- Long conversations with many tool calls
- Tool results contain large file contents
- Conversation history accumulates over time

## Lazy Tool Loading

When tools exceed `MAX_TOOLS_CHARS` (e.g., Claude Code with 36 tools = ~90k chars), the router uses lazy loading to fit within model context limits:

### How It Works

1. Model sees only tool names (not full definitions) + `search_tools` tool
2. Model calls `search_tools(["Read", "Write"])` to get specific definitions
3. Router returns definitions internally, model continues
4. This loop runs internally (invisible to client)
5. When model calls a real tool, response is returned to client

### Flow Diagram

```
Claude Code sends request (36 tools, 88k chars)
        ↓
Router detects: tools_chars > MAX_TOOLS_CHARS
        ↓
Enable lazy loading: inject tool names + search_tools
        ↓
┌─────────────────────────────────────┐
│  Internal Loop (client doesn't see) │
│  Round 1: Model calls search_tools  │
│  Round 2: Model gets definitions    │
│  Round 3: Model calls Read tool     │
└─────────────────────────────────────┘
        ↓
Return Anthropic format response:
{
  "content": [
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "toolu_xxx", "name": "Read", "input": {...}}
  ],
  "stop_reason": "tool_use"
}
```

## Project Structure

```
llm_router/
├── __init__.py        # Package exports
├── cli.py             # CLI entry point, loads .env
├── server.py          # Flask endpoints, lazy loading logic
├── mcp_converter.py   # MCP/JSON parsing, format conversion
├── llm_client.py      # OpenAI client for backend requests
├── model_config.py    # Content validation (text/multimodal)
└── debug_log.py       # Debug logging utilities
tests/
└── test_mcp_converter.py  # Unit tests
```

## Debug Mode

Enable debug logging to see full request/response details:

```bash
llm-router --debug
```

This creates `llm_router.log` with 4 types of entries:

| Log Type | Description |
|----------|-------------|
| `CLIENT_REQUEST` | Original request from client (OpenAI/Anthropic format) |
| `LLM_REQUEST` | Request sent to LLM backend (after MCP prompt injection) |
| `LLM_RESPONSE` | Raw response from LLM backend |
| `CLIENT_RESPONSE` | Final response sent to client (after tool parsing) |

Example log:
```
================================================================================
[2026-02-05 22:00:00] CLIENT_REQUEST /v1/messages
{ "model": "...", "messages": [...], "stream": true, ... }

================================================================================
[2026-02-05 22:00:00] LLM_REQUEST
{ "base_url": "...", "model": "...", "openai_params": {...}, "extra_params": {...} }

================================================================================
[2026-02-05 22:00:01] LLM_RESPONSE
{ "id": "...", "choices": [...], "usage": {...} }

================================================================================
[2026-02-05 22:00:01] CLIENT_RESPONSE /v1/messages
{ "id": "...", "content": [...], "stop_reason": "..." }
```

## Testing

```bash
pytest
pytest -v

# Or with uv
uv run pytest -v
```

## Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5001/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="MiroThinker",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }]
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

## License

MIT
