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

The main target today is Codex with DeepSeek official Chat API, explicit
third-party Responses passthrough routes, and MiroThinker MCP-first behavior.

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

- `chat`: Chat-compatible provider adapter
- `mcp_first`: MiroThinker MCP XML prompt and parser
- `responses`: generic OpenAI-compatible behavior where configured
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
reasoning replay data.

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

DeepSeek is a Chat API route with provider-specific filtering and reconstruction.
Official DeepSeek at `https://api.deepseek.com` must stay on this Chat route.

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

The router currently accepts `stream: true`, but provider payloads are forced to
non-streaming internally. If Codex requested streaming, the router returns a
simulated Responses SSE sequence:

- `response.created`
- `response.output_item.added`
- `response.output_text.delta` or `response.function_call_arguments.delta`
- `response.output_item.done`
- `response.completed`

This is enough for current item and tool-argument delivery. It is not full
Responses SSE parity.

## Logging

`--debug` enables JSONL diagnostics in:

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
