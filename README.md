# LLM Router

A Flask-based API proxy that enables standard OpenAI API clients to work with LLM backends using MCP XML format for tool calls (e.g., MiroThinker on SGLang).

## Features

- **Protocol Translation**: Accepts OpenAI API format
- **MCP Tool Conversion**: Converts API tools to MCP system prompt, parses responses back to OpenAI format
- **Dual Format Support**: Parses both MCP XML (`<use_mcp_tool>`) and JSON (`{"name": "...", "arguments": {...}}`) tool calls
- **JSON Repair**: Handles malformed JSON in tool arguments
- **vLLM/SGLang Support**: Non-standard parameters passed via `extra_body`

## Installation

```bash
# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## How It Works

### Basic Flow

```
Client (OpenAI format)
        ↓
    Tool definitions → MCP System Prompt
        ↓
    Backend LLM responds with tool calls
    (MCP XML or JSON format)
        ↓
    Router parses and converts
        ↓
Client receives standard OpenAI API response
```

### Tool Call Format Support

The router parses multiple formats from backend LLMs:

**MCP XML format**:
```xml
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>{"location": "London"}</arguments>
</use_mcp_tool>
```

**JSON format**:
```json
{"name": "get_weather", "arguments": {"location": "London"}}
```

All formats are converted to standard OpenAI format:

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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:8000` | Backend LLM URL |
| `LLM_API_KEY` | - | Backend API key (optional) |
| `FLASK_PORT` | `5001` | Router port |

## Debugging

Run with `--debug` flag to create `llm_router.log` with:
- `CLIENT_REQUEST` - Original request from client
- `LLM_REQUEST` - Request sent to backend (after MCP prompt injection)
- `LLM_RESPONSE` - Raw response from backend
- `CLIENT_RESPONSE` - Final response sent to client

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
