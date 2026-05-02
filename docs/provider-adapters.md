# Provider Adapter Guide

## Principle

Every provider has a slightly different Chat or Responses implementation. The
router should isolate those differences in provider adapters rather than
spreading compatibility code through `server.py` or `llm_client.py`.

Provider adapters decide what the upstream receives. The Responses state
machine decides what Codex receives and what gets committed to session state.

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

- commit session state
- create response IDs
- validate global pending-tool state
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

- filtering out `custom` or `web_search` tools just because Chat rejects them
- returning a provider tool call shape that Codex cannot match to its tool
  outputs
- registering pending tool calls before the full provider response is valid

Good behavior:

- convert provider-unsupported tools into a provider-accepted shape
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

## Bad Cases

Stop and redesign if a provider change:

- fixes one provider by changing global Chat behavior
- sends undocumented fields upstream
- treats model name alone as workload identity
- silently ignores multimodal, hosted-tool, or reasoning data
- bypasses the Responses state machine for `/v1/responses`
- commits state before upstream success
- makes tests pass by asserting only helper structure
