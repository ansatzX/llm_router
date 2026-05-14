import random

from llm_router.reasoning_summary import reasoning_summary_text


def test_reasoning_summary_text_includes_visible_header():
    summary = reasoning_summary_text("stable seed", rng=random.Random(2))

    assert summary.startswith("**少女折寿中**\n")
    assert len(summary.removeprefix("**少女折寿中**\n")) > 0


def test_reasoning_summary_text_randomly_picks_quotes():
    first = reasoning_summary_text("same seed", rng=random.Random(2))
    second = reasoning_summary_text("same seed", rng=random.Random(19))

    assert first != second


def test_reasoning_summary_text_ignores_seed_when_picking_quote():
    assert reasoning_summary_text("first seed", rng=random.Random(2)) == reasoning_summary_text(
        "second seed",
        rng=random.Random(2),
    )


def test_reasoning_summary_text_returns_empty_when_turn_needs_follow_up():
    assert reasoning_summary_text("stable seed", will_stop=False) == ""
