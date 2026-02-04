# LLM Router

A Flask-based API proxy that enables standard OpenAI/Anthropic API clients to work with LLM backends using MCP XML format for tool calls (e.g., MiroThinker on SGLang).

## Features

- **Protocol Translation**: Accepts OpenAI and Anthropic API formats
- **MCP Tool Conversion**: Converts API tools to MCP system prompt, parses `<use_mcp_tool>` responses back to standard formats
- **Lazy Tool Loading**: When tools exceed context limit, uses progressive loading via `search_tools`
- **Think Tag Handling**: Strips `<think>` reasoning tags from model output
- **Content Validation**: Intercepts media content based on model capabilities (text vs multimodal)
- **JSON Repair**: Handles malformed JSON in tool arguments

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
# Start router
export LLM_BASE_URL="http://your-llm-backend:8000"
llm-router

# Or with custom port
FLASK_PORT=8080 llm-router

# Or with command line options
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

1. Client sends request with tools in OpenAI/Anthropic format
2. Router generates MCP system prompt and injects it into messages
3. Request forwarded to backend LLM (without tools parameter)
4. Backend responds with `<use_mcp_tool>` XML in content
5. Router parses XML and converts to standard API tool_use format
6. Client receives standard OpenAI/Anthropic response

### MCP XML Format

The backend LLM uses this format for tool calls:

```xml
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>{"location": "London"}</arguments>
</use_mcp_tool>
```

This gets converted to:

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
| `MAX_TOKENS_CAP` | `4096` | Max completion tokens cap |
| `MAX_TOOLS_CHARS` | `40000` | Threshold for lazy tool loading |
| `MAX_TOOL_SEARCH_ROUNDS` | `20` | Max rounds for search_tools loop |
| `LLM_REQUEST_TIMEOUT` | - | Request timeout in seconds |

## Lazy Tool Loading

When tools exceed `MAX_TOOLS_CHARS`, the router uses lazy loading:

1. Model sees only tool names (not full definitions) + `search_tools`
2. Model calls `search_tools(["Read", "Write"])` to get definitions
3. Router returns definitions, model continues
4. This loop runs internally (invisible to client)
5. When model calls a real tool, response is returned to client

```
Client: "Read file.txt"
    ↓
Router: Inject tool names + search_tools
    ↓
┌─────────────────────────────────────┐
│  Internal Loop (client doesn't see) │
│  Round 1: Model calls search_tools  │
│  Round 2: Model calls Read          │
└─────────────────────────────────────┘
    ↓
Client: Receives Read tool call
```

## Project Structure

```
llm_router/
  __init__.py       # Package entry, CLI
  server.py         # Flask app, API endpoints
  llm_client.py     # HTTP client for backend
  mcp_converter.py  # MCP XML conversion
  model_config.py   # Content validation
tests/
  test_mcp_converter.py  # Unit tests
```

## Testing

```bash
pytest
pytest -v
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
