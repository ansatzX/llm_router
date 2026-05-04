# Codex Runtime Rules

This document records behavior verified from local Codex source and real router
work. It should contain current facts, not future plans.

## Responses Request Shape

Normal Codex Responses requests can include fields such as:

- `model`
- `instructions`
- `input`
- `tools`
- `tool_choice`
- `parallel_tool_calls`
- `reasoning`
- `store`
- `stream`
- `include`
- `service_tier`
- `prompt_cache_key`
- `text`
- `client_metadata`

Confirmed non-findings for the current Responses builder:

- Codex does not send `repetition_penalty`.
- Codex does not send `top_k`.
- Codex does not normally send `temperature` or `top_p` on this path.

If those fields appear upstream, they came from router/provider configuration or
another compatibility layer.

## `client_metadata`

`client_metadata` is request metadata, not a reasoning or sampling parameter.
It can carry installation identity, window identity, turn metadata, and
subagent lineage such as:

- `x-codex-installation-id`
- `x-codex-window-id`
- `x-codex-turn-metadata`
- `x-openai-subagent`

Provider adapters decide whether this field is legal upstream. DeepSeek Chat
does not accept it, so it must be filtered before the provider request.

## Collaboration Modes

Codex collaboration mode is carried in instructions/developer text, not as a
separate transport field.

Important observed markers include:

- `# Collaboration Mode: Default`
- `# Plan Mode`
- `# Plan Mode (Conversational)`

For mode detection, current-turn `instructions` should take precedence over
older history text.

Codex should own client policy and tool permissions. The router only performs
narrow compatibility recovery for known provider/Codex mismatch cases.

Current router recovery:

- retry Plan-mode plain-text questions as `request_user_input`
- steer obvious Plan-mode mutation attempts toward `<proposed_plan>`

Do not turn the router into a full shell-policy engine.

## Fast And Flex

`Fast` and `Flex` are service-tier concepts, not collaboration modes. Fast mode
can appear as:

```json
{"service_tier": "priority"}
```

This is OpenAI-specific unless a provider documents compatible behavior.
Adapters decide whether to forward, translate, or drop it.

## Background Memory Workloads

Codex memory writing can use fixed default model names, but normal user turns
can also use those names. Model name alone is not enough to identify a memory
request.

Known memory defaults:

- Phase 1 extraction: `gpt-5.4-mini`
- Phase 2 consolidation: `gpt-5.4`

Codex supports config overrides for these memory models.

Phase 2 consolidation has clean request markers:

- `x-openai-memgen-request: true`
- `x-openai-subagent: memory_consolidation`

Phase 1 extraction may need prompt-signature detection:

- `## Memory Writing Agent: Phase 1 (Single Rollout)`
- `Analyze this rollout and produce JSON with raw_memory, rollout_summary, and rollout_slug`

Router rule:

- only rewrite memory models after identifying a memory workload
- do not rewrite normal user `gpt-5.4` requests based on model name alone

## Realtime Defaults

Codex realtime has separate model defaults such as:

- `gpt-realtime-1.5`
- `gpt-4o-mini-transcribe`

These are not normal `/v1/responses` text-chat requests. Do not blindly alias
them to DeepSeek chat models.

## Tool Semantics

Codex can send Responses tools that are not native Chat `function` tools, such
as `custom`, and hosted Responses tools such as `web_search`.

Router rule:

- preserve tool semantics when adapting to a provider
- convert unsupported provider tool shapes explicitly
- handle unsupported hosted tools at the provider adapter boundary instead of
  letting provider errors abort Codex turns
- do not silently drop tools to make a provider request pass validation

For DeepSeek official Chat API, all upstream tools must end up as
`tools[].type == "function"`.

## Stateful Responses Rules

The router must maintain state across `previous_response_id`.

Confirmed rules:

- assistant tool calls create pending tool-call state
- matching tool outputs satisfy pending tool calls
- unknown tool outputs are rejected before upstream calls
- duplicate tool-call IDs are rejected
- partial parallel tool outputs are rejected
- side-channel messages can appear in a batch, but pending calls must be
  satisfied before the next model turn
- upstream failure must not commit partial state

## Current Non-Targets

Do not prioritize these unless real logs show Codex calling them through this
router:

- `/v1/responses/compact`
- `/v1/memories/trace_summarize`

They exist in Codex/OpenAI contexts, but they are not current DeepSeek/router
improvement targets.
