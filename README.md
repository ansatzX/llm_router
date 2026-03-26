# LLM Router

A production-ready Flask-based API proxy that enables standard OpenAI API clients to work with LLM backends using MCP XML format for tool calls (e.g., MiroThinker on SGLang).

## Features

### Core Capabilities

#### Protocol Translation
Transparently converts between OpenAI API and MCP XML formats.

**Motivation**: Standard OpenAI clients (e.g., OpenAI Python SDK, LangChain) expect OpenAI API format. However, models like MiroThinker use MCP XML format for tool calls. This translation layer allows existing clients to work with MCP-compatible backends without any code changes.

#### MCP Tool Conversion
Converts OpenAI tools to MCP system prompt, parses responses back to OpenAI format.

**Motivation**: MCP (Model Context Protocol) requires tools to be described in the system prompt using XML-style tags. The router handles bidirectional conversion: OpenAI tool definitions -> MCP system prompt, MCP XML tool calls -> OpenAI tool_calls format.

#### Multi-Format Parsing
Supports MCP XML (`<use_mcp_tool>`), TOOL_CALL XML, and JSON tool call formats.

**Motivation**: Different models and versions may output tool calls in different formats. Supporting multiple formats increases compatibility and reduces the need for model-specific adapters. The router tries parsers in order: MCP XML -> TOOL_CALL XML -> JSON.

#### JSON Auto-Repair
Automatically repairs malformed JSON in tool arguments using `json-repair`.

**Motivation**: Models frequently generate JSON with syntax errors (missing quotes, unescaped characters, trailing commas). Without auto-repair, these errors would cause tool execution to fail. Auto-repair increases success rates without requiring perfect model output.

#### Smart Tool Name Resolution
Intelligently resolves tool names by combining server_name and tool_name from MCP XML.

**Motivation**: MCP format includes both `server_name` and `tool_name`, which may need to be combined to form the complete tool name. For example, `server_name=AskUserQuestion` + `tool_name=AskUserQuestion` might resolve to `AskUserQuestion`. The router uses heuristics to find the correct match in available tools.

#### Tool Name Caching
Optimizes repeated tool name lookups with built-in caching.

**Motivation**: Tool name resolution may require searching through all available tools. In conversations with many tool calls, this linear search becomes expensive. Caching reduces resolution from O(n) to O(1) for repeated calls with the same server_name/tool_name pair.

#### Enhanced Regex Engine
Uses `regex` library for better Unicode and pattern matching support.

**Motivation**: Standard Python `re` library has limitations with Unicode characters (e.g., Chinese, Japanese) and advanced pattern matching. The `regex` library provides better support for internationalized content and more reliable parsing across diverse inputs.

### Advanced Features

#### Schema Validation
Validates tool arguments against JSON schema during parsing.

**Motivation**: Previously, the parser only checked JSON syntax. If arguments were syntactically correct but violated the schema (e.g., missing required fields, unexpected parameters), the tool would fail at execution time. Schema validation catches these errors during parsing, enabling Rollback to trigger and allowing the model to retry with correct parameters.

**What it checks**:
- Required fields are present
- No unexpected fields (when `additionalProperties: false`)
- Basic structure matches schema

**Impact**: Transforms "parse success -> tool execution failure" scenarios into "parse failure -> Rollback -> retry", significantly improving success rates for complex tools.

#### Rollback Mechanism
Automatically retries tool calls with error feedback when parsing fails.

**Motivation**: Models may generate malformed MCP XML or invalid arguments. Without Rollback, these errors propagate to the client, which must implement its own retry logic. Server-side Rollback handles errors transparently by providing error feedback to the model and allowing it to retry within the same request, reducing client complexity and improving user experience.

**How it works**:
1. Model generates malformed tool call
2. Parser detects error (syntax or schema validation)
3. Router adds error message to conversation
4. Model retries with corrected format
5. Success or repeat (up to MAX_ROLLBACK_RETRIES)

#### vLLM/SGLang Compatibility
Non-standard parameters passed via `extra_body`.

**Motivation**: vLLM and SGLang support additional parameters not in the OpenAI spec (e.g., `repetition_penalty`, `top_k`). The router separates standard and non-standard parameters, passing non-standard ones via `extra_body` to ensure compatibility.

#### Comprehensive Logging
Debug mode logs complete request-response cycle with detailed metadata.

**Motivation**: Debugging tool call issues requires visibility into each transformation step. The debug log captures: original request, MCP prompt injection, raw backend response, parsing results, and final client response. This end-to-end traceability accelerates issue diagnosis.

#### Reasoning Content Support
Parses tool calls from both `content` and `reasoning_content` fields.

**Motivation**: Some models (like MiroThinker) may place tool calls in `reasoning_content` during thinking/reasoning phases. The router checks both fields to ensure no tool calls are missed.

#### Content Size Limits
Protects against memory issues with 10MB parsing limit.

**Motivation**: Malicious or buggy requests could include extremely large content, causing memory exhaustion. The 10MB limit prevents DoS scenarios while accommodating legitimate use cases.

## Installation

```bash
# Using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Quick Start

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
| `/v1/chat/completions` | POST | OpenAI Chat Completions API |
| `/v1/models` | GET | List available models from backend |
| `/health` | GET | Health check endpoint |
| `/` | GET | API information and available endpoints |

## Architecture

### Request Flow

```
┌─────────────────┐
│  Client Request │  OpenAI format with tools
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         LLM Router (Flask)              │
│                                         │
│  1. Extract tools from request          │
│  2. Generate MCP system prompt          │
│  3. Inject prompt into messages         │
│  4. Force stream=false (if tools)       │
│                                         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│      Backend LLM (MiroThinker)          │
│                                         │
│  Returns MCP XML tool calls:            │
│  <use_mcp_tool>                         │
│    <server_name>...</server_name>       │
│    <tool_name>...</tool_name>           │
│    <arguments>{...}</arguments>         │
│  </use_mcp_tool>                        │
│                                         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         LLM Router (Parser)             │
│                                         │
│  1. Parse MCP XML / JSON tool calls     │
│  2. Validate against schema             │
│  3. Resolve tool names intelligently    │
│  4. Auto-repair malformed JSON          │
│  5. Convert to OpenAI format            │
│                                         │
│  If parsing fails:                      │
│  6. Add error feedback to messages      │
│  7. Retry (Rollback mechanism)          │
│                                         │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Client Response │  OpenAI format with tool_calls
└─────────────────┘
```

### Tool Call Format Support

The router parses multiple formats from backend LLMs:

**1. MCP XML Format** (Primary)
```xml
<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>{"location": "London"}</arguments>
</use_mcp_tool>
```

**2. TOOL_CALL XML Format**
```xml
[TOOL_CALL]
<tool_name>get_weather</tool_name>
<arguments>{"location": "London"}</arguments>
[/TOOL_CALL]
```

**3. JSON Format**
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

## Key Features in Detail

### 1. Smart Tool Name Resolution

When MCP XML includes a `server_name`, the router intelligently resolves the complete tool name:

```python
# MCP XML
<server_name>AskUserQuestion</server_name>
<tool_name>AskUserQuestion</tool_name>

# Resolution Logic:
# - If server_name == "default" or "tools": use tool_name as-is
# - Otherwise: search in available tools for matching name
# - Cache results for performance
```

**Example**:
- `server_name=tools`, `tool_name=get_weather` → `get_weather`
- `server_name=AskUserQuestion`, `tool_name=AskUserQuestion` → searches tools for match

### 2. JSON Auto-Repair

Automatically repairs malformed JSON in tool arguments:

```python
# Malformed JSON from model
{"location": "London, "unit": "celsius}  # Missing quote, missing closing quote

# Auto-repaired to
{"location": "London", "unit": "celsius"}
```

**Repair Coverage**:
- Missing quotes
- Extra commas
- Mismatched brackets
- Common syntax errors

### 3. Schema Validation

Validates tool arguments against JSON schema during parsing:

```python
# Tool definition with schema
{
  "name": "AskUserQuestion",
  "parameters": {
    "type": "object",
    "properties": {
      "questions": {"type": "array", ...}
    },
    "required": ["questions"],
    "additionalProperties": false
  }
}

# Model output (WRONG)
{"question": "What would you like?"}

# Schema validation catches:
# - Missing required field 'questions'
# - Unexpected field 'question'
# Triggers Rollback instead of tool execution failure
```

### 4. Rollback Mechanism

Automatic retry when tool call parsing fails:

```
Attempt 1: Model generates malformed tool call
    ↓
Parser detects error (syntax or schema)
    ↓
Router adds error feedback to messages
    ↓
Attempt 2: Model generates corrected tool call
    ↓
Success!
```

**Configuration**:
```bash
# Enable Rollback (default)
MAX_ROLLBACK_RETRIES=3

# Disable Rollback
MAX_ROLLBACK_RETRIES=0
```

**Metadata in response**:
```json
{
  "choices": [...],
  "_metadata": {
    "rollback_attempts": 2,
    "rollback_success": true
  }
}
```

## Debugging

Run with `--debug` flag to create `llm_router.log`:

```bash
llm-router --debug
```

**Log Sections**:

1. **CLIENT_REQUEST** - Original request from client
   - Endpoint, method, full request data
   - Tool definitions
   - Message previews

2. **LLM_REQUEST** - Request sent to backend
   - MCP system prompt injected
   - Tools converted
   - Stream settings

3. **LLM_RESPONSE** - Raw response from backend
   - MCP XML tool calls
   - Token usage
   - Finish reason

4. **CLIENT_RESPONSE** - Final response to client
   - OpenAI format tool calls
   - Conversion status
   - Warnings/errors

5. **ROLLBACK_RETRY** - Rollback attempt details
   - Retry count
   - Parse errors
   - Response preview

**Example Log Output**:
```
================================================================================
[2026-03-26 22:47:43] ROLLBACK_RETRY
{
  "retry_count": 2,
  "max_retries": 3,
  "parse_errors": [
    "Missing required parameter 'questions' for tool 'AskUserQuestion'",
    "Unexpected parameter 'question' for tool 'AskUserQuestion'"
  ]
}

================================================================================
[2026-03-26 20:53:42] CLIENT_RESPONSE
{
  "status": "success",
  "model": "MiroThinker",
  "has_tool_calls": true,
  "tool_calls_count": 1,
  "rollback_attempts": 2,
  "usage": {"prompt_tokens": 100, "completion_tokens": 50}
}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:8000` | Backend LLM URL |
| `LLM_API_KEY` | - | Backend API key (optional) |
| `FLASK_PORT` | `5001` | Router port |
| `MCP_SERVER_NAME` | `tools` | Default MCP server name |
| `LLM_REQUEST_TIMEOUT` | - | Request timeout in seconds (optional) |
| `MAX_ROLLBACK_RETRIES` | `3` | Maximum Rollback retries (0 to disable) |

## Example Usage

### Basic Example

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
    # Output:
    # Tool: get_weather
    # Args: {"location": "London"}
```

### With MiroThinker Backend

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5001/v1",
    api_key="not-needed"
)

# MiroThinker returns tool calls in MCP XML format
response = client.chat.completions.create(
    model="MiroThinker",
    messages=[
        {"role": "user", "content": "Search for recent papers on quantum computing"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }]
)

# Router automatically converts MCP XML to OpenAI format
for tool_call in response.choices[0].message.tool_calls:
    print(f"Tool: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=llm_router

# Run specific test file
pytest tests/test_xml_parser.py -v
```

## Troubleshooting

### Common Issues

**1. Backend returns 404**
```
Solution: Check LLM_BASE_URL is correct
- Ensure backend is running
- Verify URL includes correct port
- Check if backend uses /v1 prefix or not
```

**2. Tool calls not parsed**
```
Solution: Enable debug mode
- Run with --debug flag
- Check LLM_RESPONSE in log
- Verify tool call format matches expected patterns
```

**3. JSON parsing errors**
```
Solution: Check log for repair attempts
- Router auto-repairs JSON
- Check if arguments contain unescaped quotes
- Look for warnings in CLIENT_RESPONSE
```

**4. Tool name mismatch**
```
Solution: Verify server_name handling
- Check if server_name is set in MCP XML
- Verify tool names match available tools
- Check resolved names in debug log
```

**5. Schema validation failures**
```
Solution: Check tool parameter schema
- Verify required fields are present
- Check for unexpected fields (if additionalProperties: false)
- Review ROLLBACK_RETRY logs for specific errors
```

## Limitations

### Current Limitations

1. **No Streaming Support**: Tool calls require full response for parsing
   - Router forces `stream=false` when tools are present
   - Streaming tool calls planned for future release
   - See [Streaming Support](#streaming-support) for implementation roadmap

2. **Single Backend**: Currently supports one backend at a time
   - Multi-backend routing not implemented

3. **No Request Caching**: Each request is processed independently
   - Consider adding caching for repeated requests

## Roadmap

### Completed
- [x] MCP XML parsing
- [x] JSON format support
- [x] JSON auto-repair
- [x] Smart tool name resolution
- [x] Tool name caching
- [x] Enhanced regex engine
- [x] Comprehensive logging
- [x] Rollback mechanism
- [x] Schema validation

### In Progress
- [ ] Streaming support for tool calls (see [Streaming Support](#streaming-support))

### Planned
- [ ] Multi-backend routing
- [ ] Request/response caching
- [ ] Rate limiting
- [ ] Metrics and monitoring
- [ ] OpenTelemetry integration

## Streaming Support

### Current Status

The router currently **disables streaming** when tools are present to ensure accurate parsing:

```python
if tools:
    payload["stream"] = False
```

### Why Streaming is Disabled

1. **MCP XML requires complete blocks**: Need full `<use_mcp_tool>...</use_mcp_tool>` to parse correctly
2. **Partial token handling**: Model might emit incomplete tokens across chunks (e.g., `<use_mcp_too` in one chunk, `l>` in next)
3. **State management complexity**: Need to track streaming state across multiple chunks
4. **JSON repair needs complete JSON**: Can't repair partial JSON fragments
5. **Schema validation needs complete arguments**: Validation requires fully-formed arguments object

### Future Implementation

Streaming support will require:

**Phase 1: Token Buffering**
- Buffer partial tokens (e.g., `<use_mcp_` → `tool>`)
- Detect tool call start tokens
- Accumulate until complete block

**Phase 2: State Machine**
- States: `text` | `tool`
- Track partial tag prefixes
- Handle token boundaries correctly
- Example implementation from MiroThinker uses `_longest_token_prefix_at_end()` function

**Phase 3: Incremental JSON Processing**
- Stream JSON incrementally
- Repair on completion
- Validate on close
- Emit tool calls as they complete

**Reference Implementation**
See [MiroThinker's vLLM Tool Parser](https://github.com/MiroMindAI/MiroThinker/blob/main/apps/lobehub-compatibility/MiroThinkerToolParser.py) for streaming implementation details (search for `extract_tool_calls_streaming` method).

**Estimated Complexity**: High (3-5 weeks development + testing)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **MiroMind AI** - [MiroThinker](https://github.com/MiroMindAI/MiroThinker) - MCP format specification
- **vLLM** - Tool parser architecture inspiration
- **json-repair** - Automatic JSON repair functionality
- **regex library** - Enhanced pattern matching

## Related Projects

- [MiroThinker](https://github.com/MiroMindAI/MiroThinker) - MCP-compatible LLM
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [SGLang](https://github.com/sgl-project/sglang) - Structured generation language

---

Built for the AI community
