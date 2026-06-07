# LLM Router

`llm-router` is a local Codex-facing Responses compatibility runtime. It
accepts Codex/OpenAI-style `/v1/responses` traffic, keeps local response state
for router-owned routes, adapts requests to provider Chat APIs, and returns
Responses-shaped output to Codex.

Codex still owns local tools. The router only owns protocol adaptation,
provider routing, and router-side Responses state.

## Quick Start

```bash
uv sync
export DEEPSEEK_API_KEY="sk-..."
export MIMO_API_KEY="sk-..."        # only needed for mimo-* models
```

First-time Codex setup also needs the shared provider entry in
`~/.codex/config.toml`. Merge it from
[`codex.config.example.toml`](codex.config.example.toml):

```toml
[model_providers.llm_router]
name = "llm_router"
base_url = "http://127.0.0.1:9876/v1"
wire_api = "responses"
requires_openai_auth = false
```

`llm-router serve` syncs root-level `*.config.toml` profiles and `*.json`
model catalogs into `${CODEX_HOME:-~/.codex}`. Existing files with the same
names are overwritten so the Codex presets match this checkout.

Then launch Codex through the synced profile:

```bash
uv run llm-router serve
codex -p llm_router
```

If the default port is busy:

```bash
uv run llm-router serve --port 9877
```

Then update the matching `base_url` in `~/.codex/config.toml` to
`http://127.0.0.1:9877/v1`.

Use debug mode while diagnosing requests:

```bash
uv run llm-router serve --debug
```

Debug logs go to `llm_router.jsonl` and may contain prompts, tool payloads,
file paths, and provider metadata.

## Codex Profiles

This repo uses Codex profile-v2's three-file mode:

| File | Role |
| --- | --- |
| `codex.config.example.toml` | Shared base config example. Defines `[model_providers.llm_router]` and `[model_providers.aihubmix]`. |
| `<profile>.config.toml` | Loaded by `codex -p <profile>`. Selects `model_provider`, `model`, `model_catalog_json`, reasoning, personality, and stable runtime knobs. |
| `<profile>.json` | Model catalog loaded from `model_catalog_json`. It cannot be inlined into TOML. |

Current profiles:

| Command | Provider | Catalog | Default model |
| --- | --- | --- | --- |
| `codex -p llm_router` | local router at `127.0.0.1:9876` | `~/.codex/llm_router.json` | `deepseek-v4-pro` |
| `codex -p aihubmix` | direct AIHubMix compatibility provider | `~/.codex/aihubmix.json` | `gpt-5.5` |

Do not use legacy `[profiles.<profile>]` tables for these presets. Current
Codex resolves `-p llm_router` as `$CODEX_HOME/llm_router.config.toml`.

For direct `codex -p aihubmix`, merge the `aihubmix` provider entry from the
example config and set `AIHUBMIX_API_KEY`.

MiMo is not a separate profile. Use `codex -p llm_router` and choose a
`mimo-*` model from the `llm_router` catalog.

Catalog field rules live in
[`docs/codex-profile-catalog.md`](docs/codex-profile-catalog.md).

## Routing

Routes are configured in [`router.toml`](router.toml). First match wins.

| Model pattern | Type | Upstream | Meaning |
| --- | --- | --- | --- |
| `deepseek-*` | `responses_chat` | `deepseek` | Official DeepSeek Chat API with router-owned Responses state. |
| `mimo-*` | `responses_chat` | `xiaomi` | Xiaomi MiMo Chat API with router-owned Responses state. |
| `mirothinker-*` | `mcp_first` | `mirothinker` | MCP/XML tool prompting and parsing. |
| fallback | `responses_chat` | `deepseek` | Unmatched text models route to DeepSeek. |

Use `responses_passthrough` only for an explicitly configured upstream that
owns a compatible native `/v1/responses` state machine. Official DeepSeek at
`https://api.deepseek.com` should stay on `responses_chat`.

## Sessions

Router-owned Responses sessions are stored at:

```text
./.llm-router/sessions.json
```

Use an explicit store path when needed:

```bash
export LLM_ROUTER_SESSION_STORE="/path/to/sessions.json"
```

Inspect or clear local sessions:

```bash
uv run llm-router status
uv run llm-router clear
uv run llm-router clear -f
```

## Tests

```bash
uv run python -m pytest -q
uv run ruff check .
```

Focused checks:

```bash
uv run python -m pytest tests/test_codex_presets.py tests/test_model_catalog.py -q
uv run python -m pytest tests/test_server_responses.py -q
uv run python -m pytest tests/test_deepseek_adapter.py -q
```

Live Codex/provider tests are opt-in because they start subprocesses and may
consume quota:

```bash
LLM_ROUTER_LIVE_CODEX_E2E=1 uv run python -m pytest tests/live -q
LLM_ROUTER_LIVE_XIAOMI=1 MIMO_API_KEY=... uv run python -m pytest tests/live/test_xiaomi_api.py -q
```

## Docs

- [`docs/architecture.md`](docs/architecture.md): runtime architecture and
  state ownership.
- [`docs/codex-profile-catalog.md`](docs/codex-profile-catalog.md): Codex
  profile JSON fields and maintenance rules.
- [`docs/codex-runtime-rules.md`](docs/codex-runtime-rules.md): verified Codex
  request/runtime facts.
- [`docs/provider-adapters.md`](docs/provider-adapters.md): provider adapter
  contract.
- [`docs/testing.md`](docs/testing.md): test strategy and commands.
- [`docs/future.md`](docs/future.md): future work only.
