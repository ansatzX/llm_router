"""Static model catalog regressions."""

import json
from pathlib import Path


def _models_by_slug() -> dict[str, dict]:
    catalog = json.loads(Path("llm_router.json").read_text(encoding="utf-8"))
    return {model["slug"]: model for model in catalog["models"]}


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
