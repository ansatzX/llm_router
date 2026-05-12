# Future Work

This document is the single place for forward-looking design notes. Other docs
should describe current behavior and verified rules.

## Priority 1: Stronger Responses SSE

Current status:

- router-owned `responses_chat` now supports real upstream Chat streaming
  translated into Responses SSE
- reasoning/text/tool-call deltas stream incrementally
- session commit remains `commit-after-success`
- mixed text+tool-call streaming in one turn is intentionally rejected by
  default, with an experimental opt-in path available

Codex consumes Responses SSE, not Chat Completion chunks. Provider streams must
be translated into Codex-compatible event ordering.

Recommended next phases:

1. Keep strict mixed-stream rejection as default safety behavior.
2. Validate the `LLM_ROUTER_EXPERIMENTAL_MIXED_STREAM=1` path with live Codex
   e2e before changing default behavior.
3. Add optional non-committed failed-turn snapshot/draft buffer for better UX.
4. TODO: track Codex parser support for `response.function_call_arguments.delta`;
   until then function tool execution depends on final `response.output_item.done`.

Important ordering rule:

- emit `response.output_item.added` before `response.output_text.delta`

DeepSeek supports SSE Chat streaming with `stream: true` and optional
`stream_options.include_usage: true`, but those chunks are Chat-shaped. They
must not be forwarded directly to Codex.

Error mapping still needs verification:

- provider disconnect before terminal chunk
- `finish_reason = "length"`
- `finish_reason = "content_filter"`
- `finish_reason = "insufficient_system_resource"`
- mixed stream failure paths with partially emitted tool/input deltas

## Priority 2: Explicit Multimodal Behavior

Codex can send image or other multimodal content. The current Chat conversion
mostly extracts text.

Future behavior should be explicit:

- support multimodal providers correctly, or
- degrade unsupported multimodal payloads in a provider-specific, observable way

Silent content dropping is not acceptable.

## Priority 3: Hosted Tool Semantics

Codex/Responses can represent hosted-tool items such as:

- hosted web search calls
- image generation calls
- tool search calls
- tool search outputs

Current DeepSeek-route behavior filters hosted tools such as `web_search`
before provider calls. The router does not implement hosted-tool execution
semantics, so this behaves like an empty/no-op hosted-tool surface for DeepSeek.

Current Xiaomi-route behavior forwards Xiaomi's provider-hosted `web_search`
tool and reconstructs a completed Responses `web_search_call` item when Xiaomi
returns search annotations. This proves Codex can observe that hosted search
occurred, but it is not a full citation UI contract. Xiaomi `web_search` turns
currently use the non-streaming upstream path even when the client asks for
SSE, so the router can reconstruct the final hosted-tool lifecycle reliably.

Future work should define unsupported behavior first, then add provider support
only when the execution and output lifecycle are clear.

TODO:

- model richer source/citation display if Codex adds a stable Responses
  annotation shape for `output_text`
- add live replay fixtures for Xiaomi stream first-packet search-source behavior
- translate Xiaomi streamed search-source chunks before enabling live upstream
  streaming for Xiaomi `web_search`
- decide whether provider-hosted search should expose query/action details when
  the upstream returns only source annotations

## Priority 4: Session Concurrency

`SessionStore` uses a JSON file, process-local locking, advisory file locking,
unique temporary files, and atomic replace. This protects the current
single-file store from same-process mutation races and fixed-temp-file
collisions across router processes.

Future work:

- replace the JSON file with a persistence layer that can merge concurrent
  commits from separate router processes
- test concurrent commits to different sessions across separate processes
- test concurrent commits to the same session
- preserve commit-after-success semantics

## Priority 5: Stable Debug Event Schema

`llm_router.jsonl` is useful, but event names and fields are still diagnostic
rather than a stable telemetry schema.

Future work:

- document event names
- keep stable keys for request/route/provider/state events
- add replay tooling for selected real logs
- redact or summarize sensitive payloads by default where possible

## Priority 6: Real Traffic Replay Tests

Many important bugs were discovered from actual Codex and provider traffic. The
normal test suite still uses mostly hand-built payloads.

Future work:

- extract small sanitized fixtures from `llm_router.jsonl`
- replay a normal text turn
- replay native function calls
- replay custom tool calls such as `apply_patch`
- replay web-search/tool wrapper behavior
- replay DeepSeek thinking-mode continuation
- replay Plan-mode recovery cases

Replay tests should be deterministic and should not call live providers.

## Priority 7: Store Semantics And Response Lifecycle

The router currently persists useful session state, but it does not implement
the full hosted Responses object model.

Future work:

- define what `store=false` means for local persistence
- separate transient turn state from durable session state more explicitly
- model `completed`, `failed`, and `incomplete` statuses consistently
- preserve output indexes where Codex depends on them
- map provider terminal errors to Responses-compatible terminal states

Do not implement broad OpenAI parity speculatively. Add lifecycle detail when a
real Codex path or provider failure requires it.

## Lower Priority / Not Current Targets

Do not prioritize these for the current DeepSeek/Codex path unless logs show
real traffic:

- `/v1/responses/compact`
- `/v1/memories/trace_summarize`

Do not implement full OpenAI Responses parity before the observed Codex path
requires it. The router should stay a practical compatibility runtime, not a
speculative clone of every hosted Responses feature.

## Design Rule For Future Work

Every future compatibility change should answer three questions:

1. What exactly does Codex send or expect?
2. What exactly does the provider document or actually accept?
3. What test or log proves the router behavior is correct?

If any answer is missing, add diagnostics or a live reproduction before
changing behavior.
