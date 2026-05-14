# Current Architecture

## Role

`llm_router` is a Codex-facing Responses compatibility runtime. It is not a
plain forwarding proxy and it is not a full OpenAI-hosted Responses clone.

Its current job is:

- accept Codex `/v1/responses` requests
- normalize Responses input items
- maintain local response/session state
- validate tool-call transitions before provider calls
- route to provider-specific Chat APIs
- adapt provider responses back into Codex-compatible Responses output
- expose enough SSE shape for current Codex streaming usage

The main target today is Codex with DeepSeek official Chat API, Xiaomi MiMo
official Chat API, explicit third-party Responses passthrough routes, and
MiroThinker MCP-first behavior.

## Main Modules

- `llm_router/server.py`: Flask routes and high-level orchestration.
- `llm_router/config.py`: route matching, upstream selection, and model rewrite.
- `llm_router/codex_recovery.py`: narrow Codex compatibility recovery and
  request/response diagnostics helpers.
- `llm_router/provider_errors.py`: provider error mapping for client-visible
  router responses.
- `llm_router/responses_state/`: local Responses state machine, SSE event
  construction, Responses tool normalization, and usage normalization.
- `llm_router/session_store.py`: JSON-backed persistent session storage.
- `llm_router/deepseek/`: DeepSeek official Chat adapter.
- `llm_router/xiaomi/`: Xiaomi MiMo official Chat adapter.
- `llm_router/mirothinker/`: MiroThinker MCP-first adapter.
- `llm_router/openai_chat.py`: generic OpenAI-compatible Chat adapter.
- `llm_router/llm_client.py`: transport wrapper around the OpenAI Python SDK.
- `llm_router/debug_log.py`: structured JSONL diagnostics.

`server.py` coordinates these pieces through small private orchestration
helpers. Provider-specific compatibility belongs in provider adapters, while
Codex client-policy recovery stays limited to the documented helpers in
`codex_recovery.py`.

## Request Flow

```text
Codex /v1/responses request
  -> route resolution from model
  -> ResponsesStateMachine.ingest_request()
  -> Codex compatibility diagnostics/recovery context
  -> provider adapter filters and converts payload
  -> upstream Chat API
  -> provider adapter parses response
  -> narrow Codex recovery retry if needed
  -> ResponsesStateMachine.commit_response()
  -> JSON or Responses SSE returned to Codex
```

For `/v1/responses`, normal router-owned routes go through local Responses
state. Route type decides provider conversion behavior:

- `mcp_first`: MiroThinker MCP XML prompt and parser
- `responses`: generic OpenAI-compatible behavior where configured
- `responses_chat`: stateful Responses semantics with Chat upstream adapter
  (for example official DeepSeek)
- `responses_passthrough`: provider-owned native Responses. The router forwards
  to upstream `/v1/responses` after light tool schema normalization and does not
  create local response state for those provider-owned response IDs.

## State Ownership

The Responses state machine owns:

- local response ID creation
- `previous_response_id` continuation
- input normalization
- pending tool-call tracking
- unknown tool-output rejection
- duplicate tool-call rejection
- partial parallel tool-output rejection
- commit-after-success semantics
- provider sidecar persistence
- current minimal SSE event construction

Provider adapters do not own global conversation state. They may own
provider-private sidecars that are stored on the session, such as DeepSeek
or Xiaomi reasoning replay data.

## Persistence

Sessions are persisted as JSON under:

```text
./.llm-router/sessions.json
```

The path is relative to the router process startup directory. Set
`LLM_ROUTER_SESSION_STORE=/path/to/sessions.json` to use an explicit store.

The store writes via atomic replace, which is acceptable for current
single-router usage. It is not yet a strong concurrent multi-process store.

Failed upstream requests must not commit input items, tool-output satisfaction,
assistant output, or provider sidecar updates.

For `responses_passthrough`, the upstream provider owns response IDs,
`previous_response_id`, and continuation state. If such a provider request
fails, the router returns a provider error instead of attempting to recover that
provider-owned response ID through the local state machine.

## Current Provider Behavior

### DeepSeek

DeepSeek uses a `responses_chat` route with provider-specific filtering and
reconstruction over the upstream Chat API. Official DeepSeek at
`https://api.deepseek.com` must stay on this Chat-adapter path.

Current DeepSeek behavior:

- drops unsupported Responses-only fields such as `client_metadata`
- maps `reasoning.effort` to Chat-compatible `reasoning_effort`
- normalizes `developer` role for Chat backends
- rewrites Codex `custom` tools into Chat `function` tools
- expands Codex `namespace` child tools into provider-visible Chat
  `function` tools
- filters unsupported hosted Responses tools such as `web_search` on DeepSeek
  routes, producing an empty/no-op provider tool surface instead of a 400
- restores wrapped custom calls as Responses output items
- restores flattened namespace child calls as Responses `function_call` items
  with `namespace` and child `name`
- persists and replays `reasoning_content` required by DeepSeek thinking mode
- exposes provider reasoning to Codex as raw reasoning content while using the
  summary channel only as a sparse Codex display hint
- replays historical namespace tool calls with the same flattened Chat function
  names that DeepSeek saw when the calls were created
- recovers persisted DeepSeek sidecars by tool `call_id` when Codex resends
  full local history without `previous_response_id`
- reports DeepSeek thinking replay failures as client-visible provider errors
  without advancing the local session

DeepSeek accepts only `tools[].type == "function"`, so tools must be converted,
not discarded.

### MiroThinker

MiroThinker is the only current `mcp_first` route.

Current MiroThinker behavior:

- native Chat tools are not forwarded
- Codex tools are converted into an MCP XML instruction prompt
- XML tool calls are parsed from content or reasoning text
- parsed calls are returned as Codex/OpenAI tool-call items
- incomplete MCP XML can trigger rollback retry
- upstream streaming is forced off for this route

### Xiaomi MiMo

Xiaomi MiMo uses a `responses_chat` route with provider-specific filtering and
reconstruction over the upstream Chat API. The default route uses the single
`xiaomi` upstream, pointed at the China Token Plan base URL
`https://token-plan-cn.xiaomimimo.com/v1`; other official Xiaomi base URLs are
kept as comments in `router.toml`.

Current Xiaomi behavior:

- preserves documented `developer` messages
- maps Codex `reasoning` / `reasoning_effort` to Xiaomi `thinking.type`
- preserves explicit provider-native `thinking` when supplied
- converts Responses `input_image` items to Chat `image_url` content parts
- preserves structured image content in tool outputs for multimodal follow-up
  turns
- converts Codex `function`, `namespace`, and `custom` tools to Chat
  `function` tools
- treats Xiaomi `web_search` as a Xiaomi-only router built-in hosted tool:
  Codex hosted search is replaced with an internal `do_web_search` function
  that the main Xiaomi model may choose to call
- restores Xiaomi `reasoning_content` as Codex reasoning output items
- persists and replays Xiaomi thinking/tool sidecars under
  `provider_state["xiaomi"]`
- runs a separate `mimo-v2-omni` search subrequest only after the model calls
  `do_web_search`, then feeds the result back as a Chat tool output
- after five consecutive internal search rounds, asks the main model through a
  tool result whether it still needs more search; if the model calls
  `do_web_search` again, the router resumes search with a fresh five-round
  window
- emits a Codex-facing reasoning summary `正在多次搜索，提醒用户` when the repeated
  search guardrail is triggered
- keeps main-request thinking separate from search retrieval; only the Xiaomi
  search subrequest uses `thinking.type = "disabled"`
- returns JSON `null` as the internal tool output when Xiaomi search fails,
  with provider status and body recorded in debug logs

Audio, TTS, and video-specific Xiaomi semantics are not yet represented as
Codex-facing Responses behavior.

### Generic OpenAI-Compatible Chat

Generic Chat backends use allowlisted payload forwarding. Do not assume that
every OpenAI-compatible endpoint accepts Responses fields.

### Responses Passthrough

`responses_passthrough` is for explicitly configured third-party providers that
expose a compatible native `/v1/responses` endpoint. This route type is not
inferred from model names or provider URLs. Multiple DeepSeek-compatible
providers should be represented as separate named upstreams in `router.toml`.

The router still normalizes tool schemas such as `input_schema` to Responses
`parameters`, rewrites the client-facing model to `upstream_model`, and can
simulate Responses SSE for non-streaming upstream calls. It does not validate
or persist provider-owned pending tool state locally.

## Streaming Today

Router-owned `responses_chat` routes can translate upstream Chat SSE into
Codex-compatible Responses SSE for supported text, reasoning, and accumulated
tool-call turns. If live upstream streaming is not safe for a route or request,
the router falls back to a simulated Responses SSE sequence built from the
completed provider response:

- `response.created`
- `response.output_item.added`
- `response.output_text.delta` or `response.function_call_arguments.delta`
- `response.output_item.done`
- `response.completed`

This is enough for current item, reasoning, text, and tool-argument delivery.
It is not full Responses SSE parity.

### Reasoning Summary Display

Provider `reasoning_content` is raw reasoning and is preserved in Responses
reasoning `content`. The router does not treat that raw text as a semantic
summary. It uses reasoning `summary` only as a Codex UI display hint.

For router-owned `responses_chat` routes, the router decides whether a completed
turn would make Codex stop using the same observable conditions Codex uses after
`response.completed`:

- a `function_call` or `custom_tool_call` output item means Codex needs a
  follow-up turn
- `end_turn: false` also means Codex needs a follow-up turn
- otherwise the turn is terminal

Only terminal non-tool turns receive a visible summary. The visible text starts
with `**少女折寿中**` so current Codex TUI can extract a header, followed by one
random quote from `llm_router/quotes.json`. Tool-call turns keep an empty
summary string so the Responses item shape is stable without adding UI noise.

In live upstream streaming, the stop/follow-up decision is only known after the
upstream stream finishes and the final Responses body has been reconstructed.
The router therefore streams raw reasoning deltas immediately, then emits at
most one `response.reasoning_summary_text.delta` just before
`response.output_item.done` when the turn is terminal. Non-streaming JSON and
simulated SSE carry the same final text in the reasoning item's `summary`.

## Logging

`--debug` enables JSONL diagnostics and Flask's development reloader. Normal
source edits restart the local router process automatically while the debug
server is running.

Diagnostics are written to:

```text
llm_router.jsonl
```

Logs are intended to answer:

- what request shape Codex sent
- which collaboration mode was detected
- which tools were available
- which provider/upstream was selected
- which payload keys were filtered
- which recovery path fired
- which response/tool-call shape returned

Treat logs as sensitive because they can contain prompts, file paths, and tool
payloads.
