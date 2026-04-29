# LLM Router

`llm-router` is a local OpenAI-compatible proxy for coding agents and OpenAI SDK
clients. It exposes `/v1/responses` and `/v1/chat/completions`, routes requests
by model name, and adapts provider-specific tool-calling behavior.

The current focus is:

- **DeepSeek official API** through its OpenAI-compatible Chat Completions API.
- **MiroThinker MCP-first models** that call tools by emitting MCP-style XML.

## What It Does

```
Codex / OpenAI SDK
        |
        |  OpenAI-compatible /v1/responses or /v1/chat/completions
        v
llm-router on 127.0.0.1:9876
        |
        |  route by model pattern
        v
DeepSeek API, MiroThinker server, or another OpenAI-compatible upstream
```

The router keeps Codex-facing tool execution local. It does not execute shell
commands or patches itself; it returns tool-call items to Codex, and Codex runs
the tools.

## Supported Model Routes

The default `router.toml` defines these route patterns:

| Model pattern | Route type | Upstream | Intended use |
| --- | --- | --- | --- |
| `deepseek-*` | `chat` | `deepseek` | DeepSeek official API. Codex tools are converted to Chat `function` tools. |
| `mirothinker-*` | `mcp_first` | `mirothinker` | MiroThinker models that emit MCP XML tool calls instead of native tool calls. |
| `gpt-*` | `responses` | `default` | Generic OpenAI-compatible upstream with stateful session accumulation. |
| `claude-*` | `responses` | `default` | Generic fallback route if your upstream accepts those model names. |
| unmatched | `responses` | `default` | Fallback route. |

### Route Types

| Type | Behavior |
| --- | --- |
| `chat` | Stateless request conversion. Used for DeepSeek. Responses input is converted to Chat messages, and Codex tools are forwarded as Chat `function` tools. |
| `mcp_first` | Used for MiroThinker. The router injects an MCP XML tool prompt, parses `<use_mcp_tool>` output, and converts parsed calls back to OpenAI/Codex tool calls. |
| `responses` | Stateful compatibility mode. The router persists Responses items by `previous_response_id`, rebuilds Chat history, and forwards to the configured upstream. |

## Install

```bash
uv sync
```

## Configure

The repo includes a ready-to-use [`router.toml`](router.toml). You normally do
not need to edit it. Change it only if you want a different port, upstream URL,
API key environment variable, or model route pattern.

Set your DeepSeek key:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

If you use the `default` upstream, also set the key referenced by that upstream,
for example:

```bash
export AIHUBMIX_API_KEY="sk-..."
```

## Run

```bash
uv run llm-router serve
```

With detailed request/response logs:

```bash
uv run llm-router serve --debug
```

Debug logs are written to `llm_router.log`.

Use a custom config file:

```bash
uv run llm-router -c /path/to/router.toml serve
```

## Use With Codex

This repo includes two Codex helper files:

- [`llm_router.json`](llm_router.json): model catalog for Codex. This can be
  copied directly to Codex's config directory.
- [`codex.config.example.toml`](codex.config.example.toml): example Codex
  config. Use it as a template or merge the relevant provider fields into your
  existing Codex config.

Copy the model catalog:

```bash
mkdir -p ~/.codex
cp llm_router.json ~/.codex/llm_router.json
```

Then add or merge the provider config from `codex.config.example.toml` into
`~/.codex/config.toml`. See
[`codex.config.example.toml`](codex.config.example.toml) for the exact fields.

`env_key` in the Codex example is currently not the important authentication
point. The upstream API keys are resolved by `llm-router` from `router.toml`,
for example `DEEPSEEK_API_KEY` for DeepSeek. Keep the Codex-side value as a
placeholder unless your Codex version requires something different.

Start the router before launching Codex:

```bash
export DEEPSEEK_API_KEY="sk-..."
uv run llm-router serve
codex
```

Then select a routed model name in Codex:

- `deepseek-v4-pro`
- `deepseek-chat`
- `mirothinker-...`

Exact model names are passed through to the upstream. The route pattern only
chooses which adapter and upstream to use.

## Use With the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9876/v1",
    api_key="not-needed",
)

response = client.responses.create(
    model="deepseek-v4-pro",
    input="Say hello in one sentence.",
)

print(response.output_text)
```

Chat Completions also works:

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)

print(response.choices[0].message.content)
```

## DeepSeek Behavior

DeepSeek is handled by `llm_router.deepseek`.

The adapter:

- Converts Responses messages to Chat messages.
- Maps `developer` role messages to `system`.
- Converts Codex Responses tools into DeepSeek-compatible Chat `function` tools.
- Wraps Responses-only tools such as `custom` and `web_search` as functions, so
  DeepSeek still sees every tool Codex provided.
- Restores DeepSeek Chat `tool_calls` back to Codex Responses output items.
- Preserves DeepSeek `reasoning_content` when available.
- Avoids replaying historical thinking-mode tool turns without
  `reasoning_content`, because DeepSeek rejects those requests.

## MiroThinker Behavior

MiroThinker is handled by `llm_router.mirothinker`.

The adapter:

- Does not forward native Chat `tools` to MiroThinker.
- Injects an MCP XML tool-use prompt built from the Codex tool list.
- Parses tool calls from response `content` or `reasoning_content`.
- Converts parsed MCP calls back to OpenAI/Codex `tool_calls`.
- Retries when the model emits incomplete MCP XML.

This route is intended for models that are better at MCP/XML tool calling than
native OpenAI-style function calling.

## Session Commands

Responses sessions are stored at:

```text
~/.config/llm-router/sessions.json
```

Check session state:

```bash
uv run llm-router status
```

Clear all stored sessions:

```bash
uv run llm-router clear
```

Skip the confirmation prompt:

```bash
uv run llm-router clear -f
```

Clearing sessions is useful after changing provider behavior or when a previous
conversation contains incompatible historical tool-call state.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/responses` | Responses-compatible endpoint used by Codex and OpenAI SDK clients. |
| `POST /v1/chat/completions` | Chat Completions-compatible endpoint. |
| `GET /v1/models` | Proxies model listing from the default upstream, with fallback. |
| `GET /health` | Router and default-upstream health check. |
| `GET /liveness` | Process liveness check. |
| `GET /readiness` | Default-upstream readiness check. |
| `GET /metrics` | Proxies `/metrics` from the default upstream. |

## Limits

This project is a practical compatibility router, not a complete clone of the
OpenAI Responses API state machine.

Known limits:

- DeepSeek routes are stateless `chat` routes; Codex usually sends enough
  history for this to work.
- The streaming Responses event sequence is simplified.
- Provider-specific fields are preserved only where the adapter explicitly
  supports them.
- Not every Responses API field is implemented.

## Development

Run tests:

```bash
uv run python -m pytest -q
```

Run lint:

```bash
uv run ruff check .
```

## Production

For production, use a WSGI server instead of Flask's development server:

```bash
gunicorn --bind 0.0.0.0:9876 --workers 4 'llm_router.server:create_app()'
```

## License

MIT
