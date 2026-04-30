# LLM Router

`llm-router` is a local router for using non-OpenAI models from Codex. The
current target is Codex's Responses-style traffic, with provider-specific
adapters for:

- DeepSeek official API through its OpenAI-compatible Chat API.
- MiroThinker models that prefer MCP/XML tool calls.

Codex still executes local tools. The router only adapts requests and responses
between Codex and the upstream model provider.

## Supported Routes

The default route config is in [`router.toml`](router.toml).

| Model pattern | Type | Upstream | Notes |
| --- | --- | --- | --- |
| `deepseek-*` | `chat` | `deepseek` | Main supported route. Codex tools are forwarded as Chat `function` tools. |
| `mirothinker-*` | `mcp_first` | `mirothinker` | MCP-first route. Native tools are converted to an MCP XML prompt. |
| `gpt-*` | `responses` | `default` | Fallback/default upstream route. Not the focus of this project. |
| `claude-*` | `responses` | `default` | Fallback/default upstream route. Not the focus of this project. |

## Install

```bash
uv sync
```

## Configure

Set the upstream key used by `router.toml`:

```bash
export DEEPSEEK_API_KEY="sk-..."
```

The repo includes these Codex helper files:

- [`codex.config.example.toml`](codex.config.example.toml): example Codex config
  with the `llm_router` provider/profile.
- [`llm_router.json`](llm_router.json): static model catalog for the
  `llm_router` profile.

Install the static catalog:

```bash
mkdir -p ~/.codex
cp llm_router.json ~/.codex/llm_router.json
```

Then merge the relevant provider/profile settings from
[`codex.config.example.toml`](codex.config.example.toml) into
`~/.codex/config.toml`.

The intended Codex usage is:

| Command | Provider | Model catalog | Default model |
| --- | --- | --- | --- |
| `codex` | OpenAI default | Remote OpenAI catalog | `gpt-5.5` |
| `codex -p llm_router` | Local `llm_router` provider | `~/.codex/llm_router.json` | `deepseek-v4-pro` |

This keeps normal `codex` on OpenAI while `codex -p llm_router` opts into the
local router and static model catalog.

`env_key` in the Codex example is only a Codex-side placeholder for now. Real
upstream keys are read by `llm-router` according to [`router.toml`](router.toml),
for example `DEEPSEEK_API_KEY` for DeepSeek.

## Run

Start the router:

```bash
uv run llm-router serve
```

With debug logs:

```bash
uv run llm-router serve --debug
```

Debug logs are written to `llm_router.log`.

Launch Codex through the router profile:

```bash
codex -p llm_router
```

## DeepSeek Adapter

DeepSeek support lives in `llm_router.deepseek`.

The adapter currently handles:

- Responses items to Chat messages.
- `developer` role to `system` role.
- Codex `function`, `custom`, and `web_search` tools as DeepSeek-compatible Chat
  `function` tools.
- DeepSeek Chat `tool_calls` back to Codex Responses output items.
- DeepSeek `reasoning_content` round trip when available.
- Tool-call ordering repairs when Codex inserts side-channel messages between a
  tool call and its tool output.
- DeepSeek-specific payload filtering so Responses metadata such as
  `client_metadata` is not sent to DeepSeek.

## MiroThinker Adapter

MiroThinker support lives in `llm_router.mirothinker`.

The adapter currently handles:

- MCP XML prompt injection from the Codex tool list.
- Parsing `<use_mcp_tool>` output from content or reasoning text.
- Returning parsed MCP calls as Codex/OpenAI tool calls.
- Retry feedback when emitted MCP XML is incomplete.

Only MiroThinker is intended to be MCP-first.

## Sessions

Responses sessions are stored at:

```text
~/.config/llm-router/sessions.json
```

Check session state:

```bash
uv run llm-router status
```

Clear stored sessions:

```bash
uv run llm-router clear
```

Skip confirmation:

```bash
uv run llm-router clear -f
```

Clearing sessions is useful after adapter changes or when a conversation contains
old incompatible tool-call history.

## Development

Run tests:

```bash
uv run python -m pytest -q
```

Run lint:

```bash
uv run ruff check .
```
