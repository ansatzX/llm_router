"""Responses-compatible event builders."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any


def iter_sse_events(
    response_id: str,
    output_items: list[dict[str, Any]],
    usage: dict[str, Any],
) -> Iterator[str]:
    """Yield the minimal SSE event sequence Codex needs today."""
    created = {
        "type": "response.created",
        "response": {"id": response_id},
    }
    yield f"event: response.created\ndata: {json.dumps(created)}\n\n"

    for idx, item in enumerate(output_items):
        item_event = {
            "type": "response.output_item.done",
            "output_index": idx,
            "item": item,
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(item_event)}\n\n"

    completed = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "status": "completed",
            "output": output_items,
            "usage": usage,
        },
    }
    yield f"event: response.completed\ndata: {json.dumps(completed)}\n\n"
