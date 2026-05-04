# Provider Adapter Guide

## Principle

Every provider has a slightly different Chat or Responses implementation. The
router should isolate those differences in provider adapters rather than
spreading compatibility code through `server.py` or `llm_client.py`.

Provider adapters decide what the upstream receives. For router-owned routes,
the Responses state machine decides what Codex receives and what gets committed
to session state. `responses_passthrough` is provider-owned state by explicit
route contract.

## Adding A Provider

1. Add an upstream in `router.toml`.
2. Add a route with `pattern`, `type`, `upstream`, and optional
   `upstream_model`.
3. Create a provider adapter if existing adapters do not match the provider.
4. Wire adapter selection near `_chat_adapter_for`.
5. Add focused tests for request filtering, response reconstruction, and
   `/v1/responses` behavior.

Use explicit route-level `upstream_model` for provider model-name translation.
Do not hard-code normal model aliases in the transport layer.

If a provider owns a compatible native `/v1/responses` endpoint, use an explicit
`type = "responses_passthrough"` route. Do not infer passthrough from names like
`deepseek` or from a non-official base URL.

## Adapter Contract

An adapter may handle:

- request payload allowlisting
- Responses-to-Chat message conversion
- role normalization
- parameter mapping
- tool schema conversion
- response parsing
- provider-private replay state
- provider-specific retry feedback

An adapter should not:

- commit router-owned session state
- create router-owned response IDs
- validate global pending-tool state for router-owned routes
- implement Codex collaboration policy
- inject provider parameters into unrelated providers
- drop unsupported payloads silently

## Request Filtering

Use an allowlist for Chat provider payloads. Many OpenAI-compatible endpoints
reject OpenAI Responses fields.

Known problematic fields for DeepSeek and generic Chat backends include:

- `client_metadata`
- `prompt_cache_key`
- Responses `text` config unless mapped to `response_format`
- OpenAI-only `service_tier` unless provider supports it

Do not globally add:

- `repetition_penalty`
- `top_k`
- provider-specific sampling fields

If a provider needs a field, add it in that provider adapter or in route/config
handling with tests.

## Tool Handling

Codex tools must be preserved unless a feature is explicitly unsupported and
reported.

Bad behavior:

- filtering out `custom` tools just because Chat rejects them
- wrapping hosted tools such as `web_search` without preserving hosted-tool
  execution semantics
- returning a provider tool call shape that Codex cannot match to its tool
  outputs
- registering pending tool calls before the full provider response is valid

Good behavior:

- convert provider-unsupported tools into a provider-accepted shape
- handle explicitly unsupported hosted tools before provider calls
- restore Codex-facing output items after the provider response
- keep stable `call_id` values across call and output

## Provider Sidecars

Some providers require metadata that Codex does not echo back. DeepSeek thinking
mode is the current example: multi-round requests must replay
`reasoning_content`.

Provider-private metadata should be stored under the session's
`provider_state`, not in process-global mutable state. Update this sidecar only
as part of a successful response commit.

## MiroThinker MCP-First

Only MiroThinker is currently `mcp_first`.

MCP-first means:

- do not forward native Chat tools
- inject MCP XML instructions
- parse XML tool calls from model output
- convert parsed calls back into Codex/OpenAI tool-call items
- retry incomplete XML only within the configured rollback limit

Do not apply MCP-first behavior to other providers unless provider behavior has
been verified.

## Responses Passthrough

`responses_passthrough` is a route-level contract. The provider owns response
IDs, continuation state, and provider-side pending tool bookkeeping. The router
may normalize request schema aliases and reconstruct SSE shape for Codex, but it
must not pretend provider-owned `previous_response_id` values can be recovered
through the local session store.

Official DeepSeek remains a Chat adapter target. Third-party DeepSeek-compatible
gateways with native Responses support should be separate named upstreams, for
example `deepseek_gateway` or `aicc0`.

## Bad Cases

Stop and redesign if a provider change:

- fixes one provider by changing global Chat behavior
- sends undocumented fields upstream
- treats model name alone as workload identity
- silently ignores multimodal, hosted-tool, or reasoning data
- implicitly bypasses the Responses state machine without an explicit
  `responses_passthrough` route
- commits state before upstream success
- makes tests pass by asserting only helper structure
