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
    """Yield a compact Responses SSE sequence compatible with passthrough clients."""
    created = {
        "type": "response.created",
        "response": {"id": response_id},
    }
    yield f"event: response.created\ndata: {json.dumps(created)}\n\n"

    for idx, item in enumerate(output_items):
        if item.get("type") == "message":
            added_event = {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": item,
            }
            yield f"event: response.output_item.added\ndata: {json.dumps(added_event)}\n\n"

            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    delta_event = {
                        "type": "response.output_text.delta",
                        "output_index": idx,
                        "item_id": item.get("id"),
                        "delta": content["text"],
                    }
                    yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"

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
