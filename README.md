# LLM Router

OpenAI-compatible API proxy for LLM backends using MCP XML tool calls (e.g., MiroThinker on SGLang).

```
Client (OpenAI SDK) → llm-router → SGLang (MiroThinker)
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

```bash
# Create .env file
cp .env.example .env

# Minimal config
export LLM_BASE_URL=http://localhost:8000
export FLASK_PORT=5001
```

## Run

```bash
# Development
llm-router

# Production
gunicorn --bind 0.0.0.0:5001 --workers 4 llm_router.server:app

# With debug logging
llm-router --debug
```

## Configuration

### Core

```bash
LLM_BASE_URL=http://localhost:8000  # Backend URL
FLASK_PORT=5001                     # Router port
MAX_ROLLBACK_RETRIES=3              # Tool parse retries
```

### Timeouts

```bash
LLM_CONNECT_TIMEOUT=5     # TCP connection
LLM_READ_TIMEOUT=120      # Max idle time (not total request time)
LLM_WRITE_TIMEOUT=30      # Write timeout
LLM_POOL_TIMEOUT=10       # Pool timeout
```

### Performance

```bash
GUNICORN_WORKERS=4        # Worker processes
GUNICORN_THREADS=2        # Threads per worker
GUNICORN_TIMEOUT=120      # Worker timeout
```

## Usage

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5001/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="mirothinker",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### With Tools

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker",
    "messages": [{"role": "user", "content": "Weather in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
      }
    }]
  }'
```

## Features

- **Protocol Translation**: OpenAI API ↔ MCP XML formats
- **Tool Call Parsing**: MCP XML, TOOL_CALL XML, JSON formats
- **Rollback Retry**: Auto-retry with error feedback on parse failures
- **Schema Validation**: Validates tool arguments against schema
- **Thread-Safe**: Concurrent request handling with proper locking
- **Fine-Grained Timeouts**: Separate connect/read/write/pool timeouts

## Monitoring

### Health Check

```bash
curl http://localhost:5001/health
```

### Logs

```bash
llm-router --debug  # Enable llm_router.log
```

### Check Parsing

```bash
tail -f llm_router.log | grep "Tool call parsing"
```

## Troubleshooting

**Backend connection failed**
```bash
curl http://localhost:8000/health  # Check backend
```

**Timeout issues**
```bash
LLM_READ_TIMEOUT=300     # For slow models
GUNICORN_TIMEOUT=360
```

**502 errors**
```bash
GUNICORN_TIMEOUT=300
LLM_READ_TIMEOUT=290
```

**Tool parse warnings**
```bash
MAX_ROLLBACK_RETRIES=5
```

## Testing

```bash
pytest                  # Run tests
pytest -v               # Verbose
ruff check .            # Lint
```

## License

MIT
