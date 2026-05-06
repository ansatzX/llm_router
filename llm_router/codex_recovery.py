"""Narrow Codex compatibility recovery helpers."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_router.responses_state import ResponsesStateError


def _extract_mode_markers(content: str) -> list[str]:
    """Extract collaboration-mode names from one text blob in source order."""
    return [
        match.group("collab") or match.group("simple")
        for match in re.finditer(
            r"# Collaboration Mode:\s*(?P<collab>\w+)|# (?P<simple>Plan|Default) Mode\b",
            content,
        )
    ]


def _extract_collaboration_mode(
    messages: list[dict[str, Any]],
    instructions: str = "",
) -> str | None:
    """Find the active collaboration mode, preferring current-turn instructions."""
    if isinstance(instructions, str) and instructions:
        instruction_modes = _extract_mode_markers(instructions)
        if instruction_modes:
            return instruction_modes[-1]

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        modes = _extract_mode_markers(content)
        if modes:
            return modes[-1]
    return None


def _command_looks_like_plan_execution(command: str) -> bool:
    """Detect obvious Plan-mode execution attempts without classifying all shell commands."""
    normalized = command.strip().lower()
    execution_markers = (
        "mkdir ",
        "touch ",
        "cp ",
        "mv ",
        "rm ",
        "git init",
        "uv pip install",
        "pip install",
        "python -m pip install",
        "python3 -m pip install",
        "npm install",
        "pnpm add",
        "yarn add",
        "poetry add",
        "cargo add",
        "conda install",
        "brew install",
        "apt install",
        "apt-get install",
        "uv venv",
        "python -m venv",
        "python3 -m venv",
    )
    return (
        ">" in normalized
        or ">>" in normalized
        or any(marker in normalized for marker in execution_markers)
    )


def _plan_mode_mutation_violation(
    collaboration_mode: str | None,
    output_items: list[dict[str, Any]],
) -> str | None:
    """Describe Plan-mode mutation violations, if any."""
    if collaboration_mode != "Plan":
        return None

    for item in output_items:
        item_type = item.get("type")
        if item_type == "custom_tool_call":
            return "Plan mode forbids mutating tool calls such as apply_patch."
        if item_type != "function_call":
            continue
        name = item.get("name")
        if name == "apply_patch":
            return "Plan mode forbids mutating tool calls such as apply_patch."
        if name != "exec_command":
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            continue
        command = parsed.get("cmd")
        if isinstance(command, str) and _command_looks_like_plan_execution(command):
            return "Plan mode forbids mutating exec_command calls."
    return None


def _validate_plan_mode_output_items(
    collaboration_mode: str | None,
    output_items: list[dict[str, Any]],
) -> None:
    """Reject tool calls that obviously violate Plan mode's non-mutation rule."""
    violation = _plan_mode_mutation_violation(collaboration_mode, output_items)
    if violation:
        raise ResponsesStateError(violation, "plan_mode_violation")


def _looks_like_plaintext_clarifying_question(text: str) -> bool:
    """Heuristic for Plan-mode prompts that should have used request_user_input."""
    normalized = text.strip()
    if not normalized:
        return False
    if normalized.endswith(("?", "？", ":", "：")):
        return True
    question_markers = (
        "第一个问题",
        "第二个问题",
        "第三个问题",
        "下一个问题",
        "接下来我需要",
        "我需要搞清",
    )
    return any(marker in normalized for marker in question_markers)


def _plan_mode_should_retry_with_request_user_input(
    collaboration_mode: str | None,
    output_items: list[dict[str, Any]],
    response_message: dict[str, Any],
) -> bool:
    """Detect Plan-mode plain-text questioning that should be retried via tool call."""
    if collaboration_mode != "Plan":
        return False
    if response_message.get("tool_calls"):
        return False
    if [item.get("type") for item in output_items] != ["message"]:
        return False
    content = response_message.get("content")
    if not isinstance(content, str):
        return False
    return _looks_like_plaintext_clarifying_question(content)


def _append_plan_mode_retry_feedback(
    payload: dict[str, Any],
    response_message: dict[str, Any],
) -> None:
    """Append one corrective retry turn asking the model to use request_user_input."""
    content = response_message.get("content", "")
    if content:
        payload.setdefault("messages", []).append({
            "role": "assistant",
            "content": content,
        })
    payload.setdefault("messages", []).append({
        "role": "user",
        "content": (
            "In Plan mode, clarifying questions must use the request_user_input tool "
            "instead of plain assistant text. Re-emit your latest question as exactly "
            "one request_user_input tool call."
        ),
    })


def _append_plan_mode_proposed_plan_feedback(
    payload: dict[str, Any],
    response_message: dict[str, Any],
) -> None:
    """Append one corrective retry turn asking the model to emit proposed_plan."""
    content = response_message.get("content", "")
    if content:
        payload.setdefault("messages", []).append({
            "role": "assistant",
            "content": content,
        })
    payload.setdefault("messages", []).append({
        "role": "user",
        "content": (
            "You are still in Plan mode. Do not write files, create directories, or "
            "call mutating tools. If the design is decision complete, emit the final "
            "approved plan as exactly one <proposed_plan>...</proposed_plan> block so "
            "the client can exit Plan mode and start execution. Otherwise continue "
            "planning with non-mutating actions only."
        ),
    })


def _responses_request_diagnostics(
    data: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
    model_type: str,
    upstream_name: str,
) -> dict[str, Any]:
    """Summarize request state that helps debug mode/tool-call behavior."""
    collaboration_mode = _extract_collaboration_mode(
        messages,
        data.get("instructions", ""),
    )
    request_user_input_available = None
    if collaboration_mode == "Plan":
        request_user_input_available = True
    elif collaboration_mode == "Default":
        request_user_input_available = False

    tool_names = [
        tool.get("name", tool.get("type", "unknown"))
        for tool in data.get("tools", [])
        if isinstance(tool, dict)
    ]
    return {
        "endpoint": "/v1/responses",
        "model": model,
        "model_type": model_type,
        "upstream": upstream_name,
        "collaboration_mode": collaboration_mode,
        "request_user_input_available": request_user_input_available,
        "has_previous_response_id": bool(data.get("previous_response_id")),
        "input_message_count": len(messages),
        "tool_count": len(tool_names),
        "tool_names": tool_names,
    }


def _responses_response_diagnostics(
    llm_response: dict[str, Any],
    output_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize provider response shape for debugging router decisions."""
    choice = llm_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls") or []
    return {
        "finish_reason": choice.get("finish_reason"),
        "has_tool_calls": bool(tool_calls),
        "tool_call_names": [
            tool_call.get("function", {}).get("name")
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ],
        "output_item_types": [item.get("type") for item in output_items],
    }
