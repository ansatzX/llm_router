# Codex Profile Catalog Fields

This document describes the `<profile>.json` files used by the repository's
Codex profile-v2 presets, for example `llm_router.json` and `aihubmix.json`.
Each profile is a three-file setup:

- `codex.config.example.toml` defines shared `[model_providers.<profile>]`
  entries.
- `<profile>.config.toml` selects `model_provider`, `model`, and
  `model_catalog_json = "~/.codex/<profile>.json"`.
- `<profile>.json` provides the model catalog Codex loads from
  `model_catalog_json`.

Codex currently parses `model_catalog_json` as a path to JSON and deserializes
that file as `ModelsResponse { models: Vec<ModelInfo> }`. The catalog cannot be
inlined inside `<profile>.config.toml`.

## Maintenance Rules

- Check the local Codex source before changing catalog fields. The source of
  truth is `codex-rs/protocol/src/openai_models.rs` in the maintainer-provided
  Codex checkout.
- Run `codex features list` before accepting a field/value that depends on a
  Codex feature. Do not use catalog fields that rely on experimental,
  under-development, deprecated, or removed features unless the maintainer
  explicitly approves the risk.
- Keep `experimental_supported_tools` present because current Codex `ModelInfo`
  requires it, but keep it as an empty list unless current Codex source and
  feature classification prove the requested tool path is stable.
- Do not add `tool_mode` while `code_mode` / `code_mode_only` are not stable.
- Do not add `multi_agent_version` while `multi_agent_v2` is not stable.
- Do not add `use_responses_lite` for router-owned profiles unless Codex source,
  router behavior, and focused tests prove the alternate transport is intended.
- Do not keep upstream-only metadata fields that current Codex source ignores,
  such as `prefer_websockets`, `minimal_client_version`,
  `available_in_plans`, or `reasoning_summary_format`.

## Top-Level Shape

`<profile>.json` must be a JSON object with one non-empty field:

| Field | Meaning |
| --- | --- |
| `models` | Array of Codex model catalog entries. Each entry describes one model slug Codex can select or display for this profile. |

## Model Fields

Every model entry in this repository should expose the stable field set below.
Use explicit `null`, `false`, or empty arrays when the value is intentionally
unset; this keeps defaults visible during review.

| Field | Meaning |
| --- | --- |
| `slug` | Exact model name Codex sends in requests. For `llm_router`, this is also the route key the router matches before optional upstream rewrites. |
| `display_name` | Human-readable name shown in Codex UI surfaces. |
| `description` | Short model description. Use `null` only when no useful display description exists. |
| `default_reasoning_level` | Default reasoning effort Codex should use when no profile or user override is active. |
| `supported_reasoning_levels` | List of advertised reasoning effort presets. Each item has `effort` and `description`. Keep values aligned with the provider or router adapter. |
| `shell_type` | Shell tool mode advertised for the model. Use stable values already supported by Codex, such as `shell_command`. |
| `visibility` | Picker visibility. `list` means show in model picker; `none` hides the model. |
| `supported_in_api` | Whether Codex may use this model through its API runtime. |
| `priority` | Sort/default priority among visible models. Lower values rank earlier. |
| `additional_speed_tiers` | Legacy fast-tier markers. Prefer `service_tiers` for new work; keep this explicit for compatibility. |
| `service_tiers` | Stable service-tier choices Codex may expose, for example a `priority` tier displayed as Fast. |
| `default_service_tier` | Default service tier for the model, or `null` when the user/profile must choose or use Codex defaults. |
| `availability_nux` | Optional first-use availability message, or `null` for no message. |
| `upgrade` | Optional upgrade/migration metadata, or `null` for no model upgrade prompt. |
| `base_instructions` | Base system/developer instructions Codex uses for the model. Keep this synchronized with the intended Codex personality and repo behavior. |
| `model_messages` | Optional structured model-message templates and instruction variables. Use `null` only when `base_instructions` is sufficient. |
| `supports_reasoning_summaries` | Whether Codex should expect reasoning summaries from this model path. |
| `default_reasoning_summary` | Default reasoning-summary setting, such as `none` or `auto`. |
| `support_verbosity` | Whether Codex should expose GPT-style verbosity control. |
| `default_verbosity` | Default verbosity value, or `null` if verbosity is unsupported. |
| `apply_patch_tool_type` | Apply-patch tool protocol to advertise. Use `freeform` only when current Codex source supports it for the selected model path. |
| `web_search_tool_type` | Web-search tool representation to advertise. Keep this aligned with stable Codex tool behavior and router/provider support. |
| `truncation_policy` | Tool-output truncation policy object. It contains `mode` (`bytes` or `tokens`) and `limit`. |
| `supports_parallel_tool_calls` | Whether the model/provider path can safely handle parallel tool calls. |
| `supports_image_detail_original` | Whether image inputs may request original detail. Keep false unless the model/provider path has verified support. |
| `context_window` | Effective context window Codex should plan around. |
| `max_context_window` | Maximum allowed context window for config overrides. |
| `auto_compact_token_limit` | Explicit auto-compact threshold, or `null` to let Codex derive it. |
| `effective_context_window_percent` | Percentage of context considered usable after reserving headroom. Current default is `95`. |
| `experimental_supported_tools` | Required current Codex field for model metadata. Keep `[]` for these presets unless a stable tool path is proven. |
| `input_modalities` | Input types accepted by the model path, such as `text` or `image`. |
| `supports_search_tool` | Whether Codex may use the native search tool path with this model. This is separate from router-side provider search behavior. |
| `auto_review_model_override` | Optional model override for auto-review, or `null` for normal review-model selection. |

## Verification

Run these checks after editing profile preset files:

```bash
uv run python -m pytest tests/test_codex_presets.py tests/test_model_catalog.py -q
codex features list
codex -p llm_router mcp list
codex -p aihubmix mcp list
```

For provider-facing changes, also run focused router/provider regression tests or
live probes as allowed by the repository gate.
