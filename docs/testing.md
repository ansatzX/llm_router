# Testing Guide

## Current Test Role

The test suite protects real router behavior:

- Responses session continuation
- pending tool-call validation
- provider payload filtering
- DeepSeek reasoning replay state
- Xiaomi thinking, image, web-search, and reasoning replay behavior
- MiroThinker MCP parsing and retry behavior
- model routing and upstream model rewrites
- Plan-mode compatibility recovery
- debug JSONL logging

Prefer regression tests that describe observed failures or protocol boundaries.
Avoid tests that only lock in incidental helper structure.

## Main Test Files

- `tests/test_server_responses.py`: aggregate entrypoint for the split
  `/v1/responses` regression suite. Keep this file so the focused command in
  docs and agent instructions continues to work.
- `tests/responses/basic_flow.py`: basic Responses continuation and output
  shape.
- `tests/responses/routing_and_passthrough.py`: routing, model rewrites,
  passthrough, and memory workload detection.
- `tests/responses/plan_mode.py`: Codex Plan-mode diagnostics and recovery.
- `tests/responses/state_and_deepseek.py`: session mutation, DeepSeek provider
  sidecars, and thinking-mode replay behavior.
- `tests/responses/validation_and_tools.py`: tool-output validation,
  unsupported tool handling, Xiaomi internal web-search continuation and
  repeated-search guardrails, streaming SSE shape, and MCP-first routing.
- `tests/test_deepseek_adapter.py`: DeepSeek request/response adapter behavior.
- `tests/test_xiaomi_adapter.py`: Xiaomi request/response adapter behavior.
- `tests/test_model_catalog.py`: static model catalog regressions.
- `tests/test_mirothinker_adapter.py`: MiroThinker MCP-first behavior.
- `tests/test_config.py`: routing and upstream model rewrite behavior.
- `tests/test_session_store.py`: persisted session behavior.
- `tests/test_thread_safety.py`: initialization and threading assumptions.
- `tests/live/test_codex_cli_e2e.py`: opt-in real Codex CLI smoke test.
- `tests/live/test_xiaomi_api.py`: opt-in live Xiaomi provider smoke tests.

Parser tests remain useful for the MiroThinker MCP path, but provider or
Responses bugs should usually be tested at adapter or server level.

## Commands

Run normal checks:

```bash
uv run python -m pytest -q
uv run ruff check .
```

Run focused files:

```bash
uv run python -m pytest tests/test_server_responses.py -q
uv run python -m pytest tests/responses/plan_mode.py -q
uv run python -m pytest tests/test_deepseek_adapter.py -q
```

Run live Codex e2e only when explicitly needed:

```bash
LLM_ROUTER_LIVE_CODEX_E2E=1 uv run python -m pytest tests/live -q
```

Live tests start a temporary router, call the real `codex` CLI, and consume
upstream provider quota. They must stay opt-in.

Run live Xiaomi provider smoke tests only when explicitly needed:

```bash
LLM_ROUTER_LIVE_XIAOMI=1 MIMO_API_KEY=... uv run python -m pytest tests/live/test_xiaomi_api.py -q
```

Set `MIMO_BASE_URL` to test a Token Plan cluster and `MIMO_LIVE_MODEL` to test a
specific MiMo deployment. These tests call Xiaomi directly and consume provider
quota.

Current Xiaomi regressions cover the meaningful provider boundary rather than
only helper shape:

- adapter conversion in `tests/test_xiaomi_adapter.py` for request filtering,
  `thinking` mapping, `developer` role preservation, image parts, structured
  tool-output images, Xiaomi web-search conversion, annotations, and reasoning
  replay.
- `/v1/responses` behavior in `tests/responses/validation_and_tools.py` for
  multimodal request forwarding, Xiaomi-only `do_web_search` exposure,
  main-request `thinking` preservation, internal search continuation,
  null-on-search-failure behavior, repeated-search questioning and continuation,
  reasoning-summary notification for repeated search, and function-tool
  preservation after hosted search is replaced.
- provider sidecar persistence in `tests/responses/state_and_deepseek.py` to
  keep Xiaomi `reasoning_content` replay isolated from DeepSeek state.
- catalog and route regressions in `tests/test_model_catalog.py` and
  `tests/test_config.py`.

## Test Expectations For Provider Work

Provider changes should include tests for:

- unsupported request fields being filtered
- provider-specific fields being mapped only for that provider
- Codex tools surviving provider conversion
- provider response items becoming correct Responses output items
- provider-private state being persisted only after success
- upstream failure not mutating session state
- provider-owned internal tool loops either terminating cleanly or proving their
  continuation policy with tests

If adding streaming, include event-order tests before live tests:

- `response.output_item.added` before `response.output_text.delta`
- streamed `response.output_item.added`, matching `response.output_item.done`,
  and final `response.completed.response.output[]` items use the same item ID
- terminal non-tool reasoning turns emit exactly one visible
  `response.reasoning_summary_text.delta`
- tool-call reasoning turns preserve an empty summary and do not emit a visible
  reasoning-summary delta
- router-owned status summaries such as Xiaomi repeated-search warnings stay
  before provider reasoning summaries and are not rewritten as random quotes
- accumulated final item equals non-streaming final item
- failed or truncated stream does not commit partial state

## Meaningful Failure Evidence

When fixing a real bug, preserve the failure shape in the test name or payload.

Good examples:

- unknown tool output rejected before upstream
- DeepSeek `client_metadata` filtered before Chat request
- memory model override applies only to memory workloads
- Plan-mode plain-text question retried as `request_user_input`

Weak examples:

- testing that a helper returns a dict without checking behavior
- asserting exact internal ordering where the protocol does not require it
- adding a snapshot that hides the real compatibility rule

The aggregate `tests/test_server_responses.py` is intentionally a compatibility
entrypoint, not a separate behavioral test file. Add new Responses cases to the
split modules under `tests/responses/`.
