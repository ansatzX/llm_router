"""State machine for local Responses API compatibility."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from llm_router.responses_state.storage import ResponsesSession, SessionStore
from llm_router.responses_state.validation import (
    ResponsesStateError,
    is_tool_call,
    is_tool_output,
    tool_call_id,
    validate_no_pending_before_client_message,
    validate_tool_output_known,
)


@dataclass
class ResponsesTurn:
    """Normalized state for one /v1/responses turn."""

    session: ResponsesSession
    input_items: list[dict[str, Any]]
    response_id: str
    new_input_tool_calls: list[dict[str, Any]]
    satisfied_input_call_ids: list[str]

    def to_chat_messages(self, converter) -> list[dict[str, Any]]:
        return converter([*self.session.items, *self.input_items])


class ResponsesStateMachine:
    """Owns Responses IDs, session continuation, and pending tool state."""

    def __init__(self, store: SessionStore):
        self.store = store

    def ingest_request(
        self,
        request_data: dict[str, Any],
        model: str,
    ) -> ResponsesTurn:
        session = self._get_or_start_session(
            request_data.get("previous_response_id"),
            model,
        )
        input_items = self.normalize_input(request_data.get("input", []))
        new_tool_calls, satisfied_call_ids = self._plan_input_items(
            session,
            input_items,
        )
        return ResponsesTurn(
            session=session,
            input_items=input_items,
            response_id=self.create_response_id(),
            new_input_tool_calls=new_tool_calls,
            satisfied_input_call_ids=satisfied_call_ids,
        )

    def commit_response(
        self,
        turn: ResponsesTurn,
        output_items: list[dict[str, Any]],
        provider_state_updates: dict[str, Any] | None = None,
    ) -> None:
        self._validate_output_items(turn.session, output_items)

        if turn.input_items:
            turn.session.add_items(turn.input_items)
        for item in turn.new_input_tool_calls:
            turn.session.register_tool_call(item, turn.session.response_id)
        for call_id in turn.satisfied_input_call_ids:
            turn.session.satisfy_tool_call(call_id)
        for item in output_items:
            if is_tool_call(item):
                turn.session.register_tool_call(item, turn.response_id)
            turn.session.add_output_item(item)
        if provider_state_updates:
            turn.session.provider_state.update(provider_state_updates)
        self.store.register_response_id(turn.session, turn.response_id)

    def create_response_id(self) -> str:
        return f"resp_{uuid.uuid4().hex[:12]}"

    def _get_or_start_session(
        self,
        previous_response_id: str | None,
        model: str,
    ) -> ResponsesSession:
        if previous_response_id:
            session = self.store.get(previous_response_id)
            if session is not None:
                return session
        return ResponsesSession(self.create_response_id(), model)

    def normalize_input(self, input_data: Any) -> list[dict[str, Any]]:
        if input_data is None or input_data == "":
            return []
        if isinstance(input_data, str):
            return [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_data}],
                },
            ]
        if isinstance(input_data, dict):
            return [input_data]
        if isinstance(input_data, list):
            return [item for item in input_data if isinstance(item, dict)]
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": str(input_data)}],
            },
        ]

    def _plan_input_items(
        self,
        session: ResponsesSession,
        input_items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        known_call_ids = set(session.pending_tool_calls)
        unresolved_call_ids = session.unresolved_tool_call_ids()
        new_tool_calls: list[dict[str, Any]] = []
        satisfied_call_ids: list[str] = []
        seen_new_call_ids: set[str] = set()

        for item in input_items:
            if is_tool_call(item):
                call_id = tool_call_id(item)
                if call_id:
                    if call_id in known_call_ids:
                        continue
                    if call_id in seen_new_call_ids:
                        raise ResponsesStateError(
                            f"Tool call id already exists: {call_id}",
                            "duplicate_tool_call",
                        )
                    known_call_ids.add(call_id)
                    unresolved_call_ids.add(call_id)
                    seen_new_call_ids.add(call_id)
                    new_tool_calls.append(item)
                continue

            if is_tool_output(item):
                call_id = tool_call_id(item)
                validate_tool_output_known(
                    call_id,
                    unresolved_call_ids,
                    known_call_ids,
                )
                unresolved_call_ids.discard(call_id)
                satisfied_call_ids.append(call_id)
                continue

        validate_no_pending_before_client_message(unresolved_call_ids)

        return new_tool_calls, satisfied_call_ids

    def _validate_output_items(
        self,
        session: ResponsesSession,
        output_items: list[dict[str, Any]],
    ) -> None:
        known_call_ids = set(session.pending_tool_calls)
        seen_call_ids: set[str] = set()
        for item in output_items:
            if not is_tool_call(item):
                continue
            call_id = tool_call_id(item)
            if not call_id:
                raise ResponsesStateError(
                    "Tool call item is missing call_id.",
                    "invalid_tool_call",
                )
            if call_id in known_call_ids or call_id in seen_call_ids:
                raise ResponsesStateError(
                    f"Tool call id already exists: {call_id}",
                    "duplicate_tool_call",
                )
            seen_call_ids.add(call_id)
