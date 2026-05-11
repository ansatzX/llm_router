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
    """Yield a compact Responses SSE sequence."""
    created = {
        "type": "response.created",
        "response": {"id": response_id},
    }
    yield f"event: response.created\ndata: {json.dumps(created)}\n\n"

    for idx, item in enumerate(output_items):
        if item.get("type") == "message":
            added_item = dict(item)
            added_item["content"] = [
                {key: value for key, value in content.items() if key != "text"}
                for content in item.get("content", [])
                if isinstance(content, dict)
            ]
            added_event = {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": added_item,
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
        elif item.get("type") == "reasoning":
            summary_parts = item.get("summary", [])
            content_parts = item.get("content", [])
            item_id = item.get("id")
            added_item = {
                "type": "reasoning",
                "summary": [
                    {key: value for key, value in part.items() if key != "text"}
                    for part in summary_parts
                    if isinstance(part, dict)
                ],
                "content": [
                    {key: value for key, value in part.items() if key != "text"}
                    for part in content_parts
                    if isinstance(part, dict)
                ],
            }
            if item_id:
                added_item["id"] = item_id

            added_event = {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": added_item,
            }
            yield f"event: response.output_item.added\ndata: {json.dumps(added_event)}\n\n"

            for summary_index, part in enumerate(summary_parts):
                if not isinstance(part, dict):
                    continue
                part_added_event = {
                    "type": "response.reasoning_summary_part.added",
                    "output_index": idx,
                    "item_id": item_id,
                    "summary_index": summary_index,
                    "part": {"type": part.get("type", "summary_text")},
                }
                yield (
                    "event: response.reasoning_summary_part.added\n"
                    f"data: {json.dumps(part_added_event)}\n\n"
                )
                delta = part.get("text")
                if delta:
                    summary_delta_event = {
                        "type": "response.reasoning_summary_text.delta",
                        "output_index": idx,
                        "item_id": item_id,
                        "summary_index": summary_index,
                        "delta": delta,
                    }
                    yield (
                        "event: response.reasoning_summary_text.delta\n"
                        f"data: {json.dumps(summary_delta_event)}\n\n"
                    )

            for content_index, part in enumerate(content_parts):
                if not isinstance(part, dict):
                    continue
                delta = part.get("text")
                if not delta:
                    continue
                content_delta_event = {
                    "type": "response.reasoning_text.delta",
                    "output_index": idx,
                    "item_id": item_id,
                    "content_index": content_index,
                    "delta": delta,
                }
                yield (
                    "event: response.reasoning_text.delta\n"
                    f"data: {json.dumps(content_delta_event)}\n\n"
                )
        elif item.get("type") in {"function_call", "custom_tool_call"}:
            added_item = dict(item)
            if item.get("type") == "custom_tool_call":
                added_item["input"] = ""
            else:
                added_item["arguments"] = ""
            added_event = {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": added_item,
            }
            yield f"event: response.output_item.added\ndata: {json.dumps(added_event)}\n\n"

            arguments_delta = item.get("arguments")
            event_name = "response.function_call_arguments.delta"
            if item.get("type") == "custom_tool_call":
                arguments_delta = item.get("input")
                event_name = "response.custom_tool_call_input.delta"
            if arguments_delta:
                function_event = {
                    "type": event_name,
                    "output_index": idx,
                    "item_id": item.get("id") or item.get("call_id"),
                    "call_id": item.get("call_id"),
                    "delta": arguments_delta,
                }
                yield (
                    f"event: {event_name}\n"
                    f"data: {json.dumps(function_event)}\n\n"
                )

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
