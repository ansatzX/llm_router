# LLM Router

A Flask-based API router for LLM backends, providing both **OpenAI** and **Anthropic** protocol support with automatic tool calling format conversion.

## Overview

This router acts as a proxy between LLM backends (which may output MCP XML format for tool calls) and clients expecting standard OpenAI or Anthropic protocol formats. It automatically converts between these formats, enabling seamless integration with existing OpenAI and Anthropic client libraries.

## Architecture

```
┌─────────────┐                 ┌──────────────────┐                 ┌──────────────┐
│   Client    │                 │   LLM Router     │                 │  LLM Backend │
│   (OpenAI   │ ─────────────▶  │                  │ ─────────────▶  │  (Any OpenAI │
│  or Anthropic) │  HTTP/JSON │  Protocol Proxy   │  HTTP/JSON │   compatible)│
└─────────────┘                 └──────────────────┘                 └──────────────┘
                                    │                   │
                                    │ MCP XML          │
                                    │ Format Detection │
                                    │ and Conversion   │
                                    ▼                   ▼
                            Native Protocol  ←─────  MCP XML
                            Response Format       Output
```

## Protocol Comparison

| Feature | OpenAI Protocol | Anthropic Protocol |
|---------|-----------------|---------------------|
| Endpoint | `/v1/chat/completions` | `/v1/messages` |
| Auth Header | `Authorization: Bearer <key>` | `x-api-key: <key>` or `Authorization: Bearer <key>` |
| Message Format | `{role, content, tool_calls?}` | `{role, content[]}` |
| Content Type | String (text) | Array of blocks (text, image, tool_use) |
| Tool Call Format | `message.tool_calls[]` array | `content[]` with `tool_use` blocks |
| Tool Result Format | `{role: "tool", tool_call_id, content}` | `{role: "user", content: [{type: "tool_result", ...}]}` |
| Streaming | Yes (`stream: true`) | Yes (`stream: true`) |

## Unified Interface

The router provides a unified interface `/v1/chat` that automatically detects the request format and routes it to the appropriate handler. This simplifies client implementation by supporting both OpenAI and Anthropic protocol formats through a single endpoint.

## Tool Call Format Conversion

### MCP XML Format (Input)

LLM backends may output tool calls in XML format:

```xml
<use_mcp_tool>
<server_name>my-tools</server_name>
<tool_name>get_weather</tool_name>
<arguments>
{
  "location": "London",
  "unit": "celsius"
}
</arguments>
</use_mcp_tool>
```

### OpenAI Format (Output)

Converted to OpenAI's `tool_calls` array:

```json
{
  "id": "chatcmpl-123",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "I'll check the weather for you.",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Anthropic Format (Output)

Converted to Anthropic's `tool_use` content blocks:

```json
{
  "id": "msg_123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll check the weather for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "get_weather",
      "input": {
        "location": "London",
        "unit": "celsius"
      }
    }
  ],
  "stop_reason": "tool_use"
}
```

## Installation

### Using UV (Recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Start the Router

```bash
# Default port 5001 (5000 is used by AirPlay on macOS)
llm-router

# Or specify LLM backend URL
export LLM_BASE_URL=http://localhost:8000
llm-router

# Or specify custom port
export FLASK_PORT=8000
llm-router
```

The router will start on `http://0.0.0.0:<port>` and proxy requests to the LLM backend.

### Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | LLM backend URL | `http://localhost:8000` |
| `LLM_API_KEY` | LLM API key (if required) | `None` |
| `FLASK_PORT` | Port for the router | `5001` |
| `MODEL_TYPE` | Model type: "text" or "multimodal" | "text" |
| `MAX_UPLOAD_SIZE_MB` | Maximum file size for uploads | 10 |
| `LLM_REQUEST_TIMEOUT` | Request timeout in seconds (not set = no timeout) | `None` |

### Example with Real LLM API

```bash
# Set environment variables for your LLM backend
export LLM_BASE_URL="http://10.27.130.30:32788"
export LLM_API_KEY="your-api-key"

# Run the router
llm-router
```

## Project Structure

```
llm_router/                    # 主包
├── __init__.py               # 包入口
├── server.py                 # Flask 服务器和路由
├── llm_client.py             # LLM 后端 HTTP 客户端
├── mcp_converter.py          # MCP XML 格式转换工具
└── model_config.py           # 模型配置和内容验证

examples/                     # 示例代码
tests/                        # 测试代码
pyproject.toml               # 包配置
README.md                    # 本文档
.gitignore                   # Git 忽略文件
```

## API Endpoints

- `POST /v1/chat` - 统一接口（自动检测协议格式）
- `POST /v1/chat/completions` - OpenAI 协议端点
- `POST /v1/messages` - Anthropic 协议端点
- `GET /v1/models` - 模型列表
- `GET /health` - 健康检查
- `GET /` - API 信息

## License

MIT License
