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

For DeepSeek Chat routes, Responses `namespace` tools are not provider-native
tools. Expand each namespace child function into a Chat `function` using the
same flat model-visible name that Codex uses for namespaced code-mode tools,
for example `mcp__Local_Read__analyze_image`. Keep the child tool's
description and JSON schema. When DeepSeek returns one of those flattened tool
calls, restore the Codex-facing item as a Responses `function_call` with both
`namespace` and the child `name`, such as `namespace: "mcp__Local_Read__"` and
`name: "analyze_image"`.

Historical namespace tool calls must also replay to DeepSeek with the flattened
provider-visible name. The committed Responses item can keep the Codex
namespace shape, but the Chat replay sent upstream must match the tool name
DeepSeek saw when it created the call.

## Provider Sidecars

Some providers require metadata that Codex does not echo back. DeepSeek thinking
mode is the current example: multi-round requests must replay
`reasoning_content`.

Provider `reasoning_content` is raw reasoning, not a provider-authored summary.
When the router emits Codex Responses reasoning items, keep the raw text in
`content` and use a short synthetic display string in `summary`. Codex decides
whether to show raw reasoning, but the router should not label raw provider
thinking as a semantic summary.

Provider-private metadata should be stored under the session's
`provider_state`, not in process-global mutable state. Update this sidecar only
as part of a successful response commit.

DeepSeek tool turns should always be replayed as structured Chat
`assistant.tool_calls` and `tool` messages. Include `reasoning_content` when it
is available from `provider_state`, but do not invent it and do not convert
historical tool calls or tool outputs into normal user-visible text transcripts.
Codex may also resend its full local history without `previous_response_id`; in
that case, recover DeepSeek sidecars from persisted sessions by stable tool
`call_id` before sending the Chat request upstream.
If DeepSeek rejects a continuation because thinking-mode `reasoning_content` is
missing, return a client-visible provider error and leave the session unchanged.
Do not silently downgrade thinking mode or fabricate provider reasoning.

For Xiaomi MiMo Chat routes, XiaomiChatAdapter is separate from DeepSeekChatAdapter and only shares the neutral Responses/Chat conversion base. The same reasoning replay sidecar pattern is used
under `provider_state["xiaomi"]`. Xiaomi documents `thinking.type`, not
OpenAI/Codex `reasoning_effort`, so the Xiaomi adapter maps Codex
`none`/`minimal` reasoning to `thinking.type = "disabled"` and other reasoning
efforts to `thinking.type = "enabled"`. If a request explicitly contains
`thinking`, the adapter preserves that provider-native value.

Xiaomi also documents `developer` messages, `image_url` content parts, and a
native `web_search` hosted tool. Treat Xiaomi `web_search` as a Xiaomi-only
router built-in hosted tool: replace Codex hosted search with an internal
`do_web_search` function, let the main Xiaomi model decide whether to call it,
run a separate Xiaomi search subrequest only when called, and append the result
as Chat tool output for continuation. Keep main-request `thinking` independent
from search retrieval; only the Xiaomi search subrequest should use the cheap
disabled-thinking retrieval setting. If search fails, log provider diagnostics
and return JSON `null` as the internal tool output. If the main model requests
more than five consecutive internal searches, append a tool result asking it
whether to continue; if it calls `do_web_search` again, continue searching with
a fresh five-search window. Add the Codex-facing reasoning summary
`正在多次搜索，提醒用户` when this repeated-search guardrail is triggered.

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
