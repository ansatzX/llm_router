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
```bash
export ENABLE_METRICS=true
export SG_ENABLE_METRICS=true
export MODEL_PATH=/mirothinker_v1.7_mini

curl -LsSf https://astral.sh/uv/install.sh | sh

export LLM_BASE_URL=http://localhost:9191
export FLASK_PORT=$PORT
git clone https://github.com/ansatzX/llm_router.git || exit
cd llm_router
uv venv
uv pip install .
nohup uv run gunicorn --bind 0.0.0.0:$PORT --workers 32 llm_router.server:app > llm_router.log 2>&1 &

python3 -m sglang.launch_server
    --served-model-name mirothinker_v1.7_mini
    --model-path $MODEL_PATH
    --tp $NUM_GPUS
    --host 0.0.0.0
    --port 9191
    --trust-remote-code
    --max-running-requests 50
    --enable-metrics
    --tool-call-parser qwen25
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
