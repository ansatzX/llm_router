"""Static model catalog regressions."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_CATALOGS = ("aihubmix.json", "llm_router.json")

STABLE_CODEX_MODEL_FIELDS = {
    "additional_speed_tiers",
    "apply_patch_tool_type",
    "auto_compact_token_limit",
    "auto_review_model_override",
    "availability_nux",
    "base_instructions",
    "context_window",
    "default_reasoning_level",
    "default_reasoning_summary",
    "default_service_tier",
    "default_verbosity",
    "description",
    "display_name",
    "effective_context_window_percent",
    "experimental_supported_tools",
    "input_modalities",
    "max_context_window",
    "model_messages",
    "priority",
    "service_tiers",
    "shell_type",
    "slug",
    "support_verbosity",
    "supported_in_api",
    "supported_reasoning_levels",
    "supports_image_detail_original",
    "supports_parallel_tool_calls",
    "supports_reasoning_summaries",
    "supports_search_tool",
    "truncation_policy",
    "upgrade",
    "visibility",
    "web_search_tool_type",
}

UNSTABLE_OR_ROUTER_UNSUPPORTED_MODEL_FIELDS = {
    "multi_agent_version",
    "tool_mode",
    "use_responses_lite",
}


def _models_by_slug() -> dict[str, dict]:
    catalog = json.loads(Path("llm_router.json").read_text(encoding="utf-8"))
    return {model["slug"]: model for model in catalog["models"]}


def _profile_catalog_models() -> list[tuple[str, dict]]:
    models: list[tuple[str, dict]] = []
    for filename in PROFILE_CATALOGS:
        catalog = json.loads((REPO_ROOT / filename).read_text(encoding="utf-8"))
        assert catalog["models"], f"{filename} must expose at least one model"
        models.extend((filename, model) for model in catalog["models"])
    return models


def test_profile_catalogs_expose_only_stable_codex_model_fields():
    """Profile catalogs should expose all stable fields Codex consumes."""
    for filename, model in _profile_catalog_models():
        keys = set(model)
        assert keys == STABLE_CODEX_MODEL_FIELDS, (
            f"{filename}:{model.get('slug')} has unexpected catalog keys; "
            "update docs/codex-profile-catalog.md and this allowlist from "
            "current Codex ModelInfo source if this is intentional"
        )
        assert not keys & UNSTABLE_OR_ROUTER_UNSUPPORTED_MODEL_FIELDS
        assert model["experimental_supported_tools"] == []


def test_mimo_v25_pro_catalog_supports_image_input():
    model = _models_by_slug()["mimo-v2.5-pro"]

    assert model["input_modalities"] == ["text", "image"]
    assert model["web_search_tool_type"] == "text"
    assert model["context_window"] == 1000000
    assert model["supported_in_api"] is True


def test_deepseek_flash_catalog_is_available():
    model = _models_by_slug()["deepseek-v4-flash"]

    assert model["input_modalities"] == ["text"]
    assert model["web_search_tool_type"] == "text"
    assert model["context_window"] == 1000000
    assert model["supported_in_api"] is True


def test_mimo_catalog_covers_official_web_search_models():
    models = _models_by_slug()

    for slug in (
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "mimo-v2-flash",
    ):
        assert models[slug]["web_search_tool_type"] == "text"
        assert models[slug]["supported_in_api"] is True


def test_mimo_catalog_marks_verified_image_models():
    models = _models_by_slug()

    assert models["mimo-v2.5-pro"]["input_modalities"] == ["text", "image"]
    assert models["mimo-v2.5"]["input_modalities"] == ["text", "image"]
    assert models["mimo-v2-omni"]["input_modalities"] == ["text", "image"]
    assert models["mimo-v2-pro"]["input_modalities"] == ["text"]
    assert models["mimo-v2-flash"]["input_modalities"] == ["text"]
