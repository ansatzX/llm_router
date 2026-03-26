# LM Studio + LLM Router 使用指南

## 启动服务

### 1. 启动 LM Studio
确保 LM Studio 正在运行，默认地址：`http://localhost:1234`

### 2. 启动 Router
```bash
cd D:/infer/llm_router
uv run llm-router
```

Router 将在 `http://localhost:5001` 启动

## 使用示例

### 1. 普通对话（无工具）

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [
      {"role": "system", "content": "You answer only in rhymes."},
      {"role": "user", "content": "What is your favorite color?"}
    ]
  }'
```

### 2. 工具调用

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mirothinker-1.7-mini",
    "messages": [
      {"role": "user", "content": "What is the weather in London?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }'
```

**返回结果包含标准 OpenAI 格式的 tool_calls：**
```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "...",
      "tool_calls": [{
        "id": "call_65057456",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"London\"}"
        }
      }]
    }
  }]
}
```

## 工作原理

1. **Router 接收请求**：标准的 OpenAI 格式
2. **注入 MCP 提示**：将工具定义转换为 MCP XML 格式的系统提示
3. **发送到 LM Studio**：MiroThinker 模型生成工具调用（MCP XML 或 JSON 格式）
4. **解析并转换**：Router 解析 MCP 响应，转换为标准 OpenAI tool_calls 格式
5. **返回客户端**：客户端收到标准的 OpenAI 格式响应

## 支持的工具调用格式

Router 可以解析以下格式：
- MCP XML: `<use_mcp_tool>...</use_mcp_tool>`
- Tool call XML: `[TOOL_CALL]...[/TOOL_CALL]`
- JSON: `{"name": "...", "arguments": {...}}`

## 配置

编辑 `.env` 文件：
```bash
LLM_BASE_URL=http://localhost:1234  # LM Studio 地址
LLM_API_KEY=                         # 不需要
FLASK_PORT=5001                      # Router 端口
```

## 端点

- `/v1/chat/completions` - OpenAI Chat Completions API
- `/v1/models` - 列出可用模型
- `/health` - 健康检查
