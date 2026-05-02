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

The main target today is Codex with DeepSeek official Chat API and MiroThinker
MCP-first behavior.

## Main Modules

- `llm_router/server.py`: Flask routes and high-level orchestration.
- `llm_router/config.py`: route matching, upstream selection, and model rewrite.
- `llm_router/responses_state/`: local Responses state machine.
- `llm_router/session_store.py`: JSON-backed persistent session storage.
- `llm_router/deepseek/`: DeepSeek official Chat adapter.
- `llm_router/mirothinker/`: MiroThinker MCP-first adapter.
- `llm_router/openai_chat.py`: generic OpenAI-compatible Chat adapter.
- `llm_router/llm_client.py`: transport wrapper around the OpenAI Python SDK.
- `llm_router/debug_log.py`: structured JSONL diagnostics.

`server.py` may coordinate these pieces, but provider-specific compatibility
belongs in provider adapters.

## Request Flow

```text
Codex /v1/responses request
  -> route resolution from model
  -> ResponsesStateMachine.ingest_request()
  -> provider adapter filters and converts payload
  -> upstream Chat API
  -> provider adapter parses response
  -> ResponsesStateMachine.commit_response()
  -> JSON or Responses SSE returned to Codex
```

For `/v1/responses`, all route types go through local Responses state. Route
type only decides provider conversion behavior:

- `chat`: Chat-compatible provider adapter
- `mcp_first`: MiroThinker MCP XML prompt and parser
- `responses`: generic OpenAI-compatible behavior where configured

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
~/.config/llm-router/sessions.json
```

The store writes via atomic replace, which is acceptable for current
single-router usage. It is not yet a strong concurrent multi-process store.

Failed upstream requests must not commit input items, tool-output satisfaction,
assistant output, or provider sidecar updates.

## Current Provider Behavior

### DeepSeek

DeepSeek is a Chat API route with provider-specific filtering and reconstruction.

Current DeepSeek behavior:

- drops unsupported Responses-only fields such as `client_metadata`
- maps `reasoning.effort` to Chat-compatible `reasoning_effort`
- normalizes `developer` role for Chat backends
- rewrites Codex `custom` and `web_search` tools into Chat `function` tools
- restores wrapped custom calls as Responses output items
- persists and replays `reasoning_content` required by DeepSeek thinking mode

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

## Streaming Today

The router currently accepts `stream: true`, but provider payloads are forced to
non-streaming internally. If Codex requested streaming, the router returns a
minimal simulated Responses SSE sequence:

- `response.created`
- `response.output_item.done`
- `response.completed`

This is enough for basic item delivery. It is not full Responses SSE parity.

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
