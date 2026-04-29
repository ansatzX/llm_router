"""Session store for Responses API state accumulation.

Sessions are persisted to disk as JSON. They survive server restarts.
Only explicit user action (CLI `clear`) removes them.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_STORE_PATH = Path.home() / ".config/llm-router/sessions.json"


class ResponsesSession:
    """Accumulated state for a Responses API conversation."""

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.items: list[dict[str, Any]] = []
        self.created_at = time.time()
        self.last_access = time.time()

    def add_items(self, items: Sequence[dict[str, Any]]):
        self.items.extend(items)
        self.last_access = time.time()

    def add_output_item(self, item: dict[str, Any]):
        self.items.append(item)
        self.last_access = time.time()

    def item_count(self) -> int:
        return len(self.items)

    def to_chat_messages(self, converter) -> list[dict[str, Any]]:
        """Convert accumulated items to Chat Completions messages."""
        return converter(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "model": self.model,
            "items": self.items,
            "created_at": self.created_at,
            "last_access": self.last_access,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResponsesSession:
        s = cls(d["response_id"], d.get("model", "unknown"))
        s.items = d.get("items", [])
        s.created_at = d.get("created_at", time.time())
        s.last_access = d.get("last_access", time.time())
        return s


class SessionStore:
    """LRU-like session store backed by a JSON file on disk.

    Sessions persist across restarts. Only the CLI `clear` command
    removes them — and it requires explicit user confirmation.
    """

    def __init__(
        self,
        store_path: str | Path = DEFAULT_STORE_PATH,
        max_sessions: int = 1000,
        ttl_seconds: int = 86400 * 30,  # 30 days default
    ):
        self.store_path = Path(store_path)
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, ResponsesSession] = OrderedDict()
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _ensure_dir(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path) as f:
                data = json.load(f)
            loaded = 0
            for rid, raw in data.get("sessions", {}).items():
                session = ResponsesSession.from_dict(raw)
                self._store[rid] = session
                loaded += 1
            if loaded:
                pass  # Silent load
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self):
        self._ensure_dir()
        # Write to temp file first, then atomically rename
        tmp = self.store_path.with_suffix(".tmp")
        unique_sessions = []
        seen_ids = set()
        for session in self._store.values():
            object_id = id(session)
            if object_id not in seen_ids:
                unique_sessions.append(session)
                seen_ids.add(object_id)
        data = {
            "version": 1,
            "updated_at": time.time(),
            "sessions": {
                s.response_id: s.to_dict() for s in unique_sessions
            },
        }
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.store_path)

    # ── Eviction ───────────────────────────────────────────────────────

    def _evict_expired(self):
        now = time.time()
        expired = [
            rid for rid, s in self._store.items()
            if now - s.last_access > self.ttl_seconds
        ]
        if expired:
            for rid in expired:
                del self._store[rid]
            self._save()

    def _evict_lru(self):
        while len(self._store) >= self.max_sessions:
            self._store.popitem(last=False)

    # ── Public API ─────────────────────────────────────────────────────

    def create(self, model: str) -> ResponsesSession:
        self._evict_expired()
        self._evict_lru()
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        session = ResponsesSession(response_id, model)
        self._store[response_id] = session
        self._save()
        return session

    def get(self, response_id: str) -> ResponsesSession | None:
        self._evict_expired()
        session = self._store.get(response_id)
        if session:
            session.last_access = time.time()
        return session

    def get_or_create(
        self, response_id: str | None, model: str,
    ) -> ResponsesSession:
        if response_id:
            session = self.get(response_id)
            if session:
                return session
        return self.create(model)

    def add_items(
        self,
        session: ResponsesSession,
        items: Sequence[dict[str, Any]],
    ) -> None:
        """Append user/input items and persist the updated session."""
        session.add_items(items)
        self._save()

    def add_output_item(
        self,
        session: ResponsesSession,
        item: dict[str, Any],
    ) -> None:
        """Append assistant output and persist the updated session."""
        session.add_output_item(item)
        self._save()

    def register_response_id(
        self,
        session: ResponsesSession,
        response_id: str,
    ) -> None:
        """Make the latest response ID resolve to this session.

        Responses API clients continue a conversation with the previous
        response's ID. Keep the store keyed by the latest response ID so the
        next request resumes the accumulated session.
        """
        old_keys = [rid for rid, stored in self._store.items() if stored is session]
        for rid in old_keys:
            del self._store[rid]
        session.response_id = response_id
        session.last_access = time.time()
        self._store[response_id] = session
        self._save()

    # ── Clear ──────────────────────────────────────────────────────────

    def clear_all(self) -> int:
        """Delete all sessions. Returns count of sessions removed."""
        count = len(self._store)
        self._store.clear()
        if self.store_path.exists():
            self.store_path.unlink()
        return count

    def stats(self) -> dict[str, Any]:
        """Diagnostic info: count, total items, disk path, etc."""
        total_items = sum(s.item_count() for s in self._store.values())
        return {
            "session_count": len(self._store),
            "total_items": total_items,
            "store_path": str(self.store_path),
            "store_size_bytes": (
                self.store_path.stat().st_size if self.store_path.exists() else 0
            ),
        }

    def __len__(self) -> int:
        return len(self._store)
