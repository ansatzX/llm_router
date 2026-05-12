# Streaming Progress Summary

Date: 2026-05-12

## Current State

`llm_router` now has a substantially stronger DeepSeek official Chat streaming
adapter for Codex-facing `/v1/responses`.

Implemented:

- DeepSeek reasoning content is converted into Responses reasoning items and
  streamed as reasoning SSE deltas.
- Router-owned `responses_chat` routes can use real upstream Chat streaming for
  text, reasoning, and tool calls.
- DeepSeek tool-call streaming is accumulated and committed only after upstream
  success.
- Tool `response.output_item.added` events now include Codex-required placeholder
  fields:
  - `function_call.arguments = ""`
  - `custom_tool_call.input = ""`
- Custom tool input deltas now use `response.custom_tool_call_input.delta` on
  both live and simulated SSE paths.
- DeepSeek custom tool wrappers such as `{"input":"..."}` can emit stable input
  prefixes before the full JSON object closes.
- Mixed text+tool streaming remains rejected by default, but can be enabled with
  `LLM_ROUTER_EXPERIMENTAL_MIXED_STREAM=1`.
- Session commit semantics remain `commit-after-success`.

## Verification

Last verified commands:

```bash
uv run python -m pytest tests/responses/basic_flow.py tests/responses/validation_and_tools.py -q
uv run python -m pytest tests/test_server_responses.py -q
uv run python -m pytest -q
uv run ruff check .
```

Result:

- `173 passed, 1 skipped`
- `ruff` passed

## Remaining Boundaries

- Mixed text+tool streaming is experimental and should stay off by default until
  live Codex e2e coverage proves item routing is stable.
- `response.function_call_arguments.delta` is currently ignored by local Codex
  SSE parsing; function tool execution still depends on final
  `response.output_item.done`.
- The custom input prefix extractor covers common JSON string escapes, but more
  live DeepSeek fixtures would improve confidence around rare escape boundaries.
- No live DeepSeek provider replay fixture has been committed yet.

## Files Changed In The Commit Scope

- `llm_router/server.py`
- `llm_router/responses_state/events.py`
- `tests/responses/basic_flow.py`
- `tests/responses/validation_and_tools.py`
- `README.md`
- `docs/future.md`

This summary file is intentionally left untracked.
