# Future Work

This document is the single place for forward-looking design notes. Other docs
should describe current behavior and verified rules.

## Priority 1: Stronger Responses SSE

Current status:

- router-owned `responses_chat` now supports real upstream Chat streaming
  translated into Responses SSE
- reasoning/text/tool-call deltas stream incrementally
- terminal non-tool turns emit one late
  `response.reasoning_summary_text.delta` for Codex TUI after the router knows
  the completed turn will stop
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
- emit terminal visible reasoning-summary delta only after the upstream stream
  finishes and before the matching reasoning `response.output_item.done`

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

Current Xiaomi-route behavior treats Xiaomi `web_search` as a Xiaomi-only
router built-in hosted tool. The router exposes Codex hosted search as an
internal `do_web_search` function to the main Xiaomi model. If the model calls
it, the router runs a separate non-streaming `mimo-v2-omni` search subrequest,
feeds the result back as tool output, and asks the main model to continue. This
does not yet provide a full Codex citation UI contract. Current production
behavior also guards repeated agentic search: after five consecutive internal
searches, the router asks the main model through a tool result whether it still
needs to continue. If the model calls `do_web_search` again, search resumes with
a fresh five-search window. Codex receives a reasoning summary
`正在多次搜索，提醒用户` when that guardrail is triggered.

Future work should define unsupported behavior first, then add provider support
only when the execution and output lifecycle are clear.

TODO:

- model richer source/citation display if Codex adds a stable Responses
  annotation shape for `output_text`
- add live replay fixtures for Xiaomi `do_web_search` result and annotation
  behavior
- add live replay fixtures for Xiaomi repeated-search questioning and
  insist-to-continue behavior
- translate Xiaomi streamed search-source chunks if the router later exposes
  live search progress instead of the current built-in search completion item
- decide whether provider-hosted search should expose query/action details when
  the upstream returns only source annotations

## Priority 3.5: Large Namespace Tool Surfaces

Current Chat-backed routes intentionally expand Codex `namespace` tools into
one provider-visible Chat `function` per namespace child. This preserves MCP
and app tool semantics because Chat Completions providers do not understand the
Responses namespace object directly. Do not remove this expansion as a quick
way to reduce tool count.

Observed Codex traffic can expand a compact namespace surface into a much
larger Chat tool list, for example tens of Codex tools becoming more than one
hundred provider functions after MCP/app child tools are exposed. There is no
verified Xiaomi official numeric tool limit in the current evidence set, but a
large expanded tool surface can still increase request size, schema prompt
cost, latency, and provider-specific rejection risk.

Future options:

- add diagnostics for original tool count, expanded tool count, and approximate
  tool schema bytes
- add explicit provider policy knobs for namespace allowlists/denylists or a
  hard cap that fails visibly instead of silently dropping tools
- prototype a compressed namespace dispatcher that exposes one Chat function
  per namespace with `{name, arguments}` and lets the router restore the child
  call for Codex

The dispatcher option is not the default plan. It reduces provider tool count
but loses per-child schemas and descriptions at the model boundary, so it needs
live Codex/provider evidence before replacing full expansion.

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
