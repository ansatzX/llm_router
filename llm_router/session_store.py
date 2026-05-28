"""Session store for Responses API state accumulation.

Sessions are persisted to disk as JSON. They survive server restarts.
Only explicit user action (CLI `clear`) removes them.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

SESSION_STORE_ENV = "LLM_ROUTER_SESSION_STORE"
DEFAULT_STORE_DIR = ".llm-router"
DEFAULT_STORE_FILE = "sessions.json"


def default_store_path() -> Path:
    """Return the startup-directory-local default session store path."""
    configured = os.environ.get(SESSION_STORE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / DEFAULT_STORE_DIR / DEFAULT_STORE_FILE


class ResponsesSession:
    """Accumulated state for a Responses API conversation."""

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.items: list[dict[str, Any]] = []
        self.pending_tool_calls: dict[str, dict[str, Any]] = {}
        self.provider_state: dict[str, Any] = {}
        self.created_at = time.time()
        self.last_access = time.time()

    def add_items(self, items: Sequence[dict[str, Any]]):
        self.items.extend(items)
        self.last_access = time.time()

    def add_output_item(self, item: dict[str, Any]):
        self.items.append(item)
        self.last_access = time.time()

    def register_tool_call(
        self,
        item: dict[str, Any],
        response_id: str,
    ) -> None:
        call_id = item.get("call_id") or item.get("id")
        if not call_id:
            return
        self.pending_tool_calls[call_id] = {
            "call_id": call_id,
            "name": item.get("name", ""),
            "type": item.get("type", ""),
            "arguments": item.get("arguments", item.get("input", "")),
            "created_response_id": response_id,
            "status": "pending",
        }
        self.last_access = time.time()

    def satisfy_tool_call(self, call_id: str) -> None:
        if call_id in self.pending_tool_calls:
            self.pending_tool_calls[call_id]["status"] = "satisfied"
            self.last_access = time.time()

    def unresolved_tool_call_ids(self) -> set[str]:
        return {
            call_id for call_id, state in self.pending_tool_calls.items()
            if state.get("status") == "pending"
        }

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
            "pending_tool_calls": self.pending_tool_calls,
            "provider_state": self.provider_state,
            "created_at": self.created_at,
            "last_access": self.last_access,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResponsesSession:
        s = cls(d["response_id"], d.get("model", "unknown"))
        s.items = d.get("items", [])
        s.pending_tool_calls = d.get("pending_tool_calls", {})
        s.provider_state = d.get("provider_state", {})
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
        store_path: str | Path | None = None,
        max_sessions: int = 1000,
        ttl_seconds: int = 86400 * 30,  # 30 days default
    ):
        self.store_path = Path(store_path).expanduser() if store_path else default_store_path()
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._store: OrderedDict[str, ResponsesSession] = OrderedDict()
        self.load_error: str | None = None
        self.load_backup_path: Path | None = None
        self._load()

    # ── Persistence ────────────────────────────────────────────────────

    def _ensure_dir(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _file_lock(self):
        """Serialize writes from multiple router processes."""
        self._ensure_dir()
        with open(self.store_path.with_suffix(".lock"), "a") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _corrupt_backup_path(self) -> Path:
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        base_name = f"{self.store_path.name}.corrupt-{timestamp}"
        candidate = self.store_path.with_name(base_name)
        suffix = 1
        while candidate.exists():
            candidate = self.store_path.with_name(f"{base_name}-{suffix}")
            suffix += 1
        return candidate

    def _preserve_unreadable_store(self, exc: Exception) -> None:
        self._store.clear()
        self.load_error = f"{exc.__class__.__name__}: {exc}"
        try:
            backup_path = self._corrupt_backup_path()
            os.replace(self.store_path, backup_path)
            self.load_backup_path = backup_path
        except OSError as backup_exc:
            self.load_error = f"{self.load_error}; backup failed: {backup_exc}"

    def _clear_file_candidates(self) -> list[Path]:
        if not self.store_path.parent.exists():
            return []

        candidates = [
            self.store_path,
            self.store_path.with_name(f"{self.store_path.name}.tmp"),
        ]
        candidates.extend(
            self.store_path.parent.glob(f"{self.store_path.name}.corrupt-*"),
        )
        candidates.extend(
            self.store_path.parent.glob(f"{self.store_path.name}.*.tmp"),
        )

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and (candidate.is_file() or candidate.is_symlink()):
                unique_candidates.append(candidate)
        return sorted(unique_candidates)

    def _load(self):
        with self._lock:
            if not self.store_path.exists():
                return
            try:
                with self._file_lock():
                    if not self.store_path.exists():
                        return
                    try:
                        with open(self.store_path, encoding="utf-8") as f:
                            data = json.load(f)
                        loaded_store: OrderedDict[str, ResponsesSession] = OrderedDict()
                        for rid, raw in data.get("sessions", {}).items():
                            session = ResponsesSession.from_dict(raw)
                            loaded_store[rid] = session
                        if loaded_store:
                            pass  # Silent load
                        self._store = loaded_store
                    except (
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                        AttributeError,
                        KeyError,
                        TypeError,
                    ) as exc:
                        self._preserve_unreadable_store(exc)
            except OSError as exc:
                self.load_error = f"{exc.__class__.__name__}: {exc}"

    def _save(self):
        with self._lock, self._file_lock():
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{self.store_path.name}.",
                suffix=".tmp",
                dir=self.store_path.parent,
            )
            tmp = Path(tmp_name)
            try:
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
                with os.fdopen(fd, "w") as f:
                    fd = None
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.store_path)
            except Exception:
                if fd is not None:
                    os.close(fd)
                if tmp.exists():
                    tmp.unlink()
                raise

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
        with self._lock:
            self._evict_expired()
            self._evict_lru()
            response_id = f"resp_{uuid.uuid4().hex[:12]}"
            session = ResponsesSession(response_id, model)
            self._store[response_id] = session
            self._save()
            return session

    def get(self, response_id: str) -> ResponsesSession | None:
        with self._lock:
            self._evict_expired()
            session = self._store.get(response_id)
            if session:
                session.last_access = time.time()
            return session

    def get_or_create(
        self, response_id: str | None, model: str,
    ) -> ResponsesSession:
        with self._lock:
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
        with self._lock:
            session.add_items(items)
            self._save()

    def add_output_item(
        self,
        session: ResponsesSession,
        item: dict[str, Any],
    ) -> None:
        """Append assistant output and persist the updated session."""
        with self._lock:
            session.add_output_item(item)
            self._save()

    def save(self) -> None:
        """Persist the current in-memory sessions."""
        with self._lock:
            self._save()

    def provider_state_for_call_ids(
        self,
        provider: str,
        call_ids: set[str],
    ) -> dict[str, Any]:
        """Recover provider-private sidecar data by tool call ID.

        Codex can replay full local history without ``previous_response_id``.
        In that shape, provider-only sidecars are not present in the input, so
        recover them from recent persisted router sessions by stable call ID.
        """
        wanted = {str(call_id) for call_id in call_ids if call_id}
        if not wanted:
            return {}

        with self._lock:
            reasoning_by_call_id: dict[str, str] = {}
            seen_sessions: set[int] = set()
            sessions = sorted(
                self._store.values(),
                key=lambda session: session.last_access,
                reverse=True,
            )
            for session in sessions:
                session_key = id(session)
                if session_key in seen_sessions:
                    continue
                seen_sessions.add(session_key)

                provider_state = session.provider_state.get(provider, {})
                if not isinstance(provider_state, dict):
                    continue
                reasoning_state = provider_state.get("reasoning_by_call_id", {})
                if not isinstance(reasoning_state, dict):
                    continue

                missing = wanted - set(reasoning_by_call_id)
                for call_id in missing:
                    value = reasoning_state.get(call_id)
                    if value:
                        reasoning_by_call_id[call_id] = str(value)
                if wanted <= set(reasoning_by_call_id):
                    break

            if not reasoning_by_call_id:
                return {}
            return {"reasoning_by_call_id": reasoning_by_call_id}

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
        with self._lock:
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
        with self._lock, self._file_lock():
            count = len(self._store)
            self._store.clear()
            for candidate in self._clear_file_candidates():
                candidate.unlink(missing_ok=True)
            self.load_error = None
            self.load_backup_path = None
            return count

    def stats(self) -> dict[str, Any]:
        """Diagnostic info: count, total items, disk path, etc."""
        with self._lock:
            total_items = sum(s.item_count() for s in self._store.values())
            clear_files = self._clear_file_candidates()
            return {
                "session_count": len(self._store),
                "total_items": total_items,
                "store_path": str(self.store_path),
                "store_size_bytes": (
                    self.store_path.stat().st_size if self.store_path.exists() else 0
                ),
                "clear_file_count": len(clear_files),
                "clear_file_size_bytes": sum(
                    path.stat().st_size for path in clear_files if path.exists()
                ),
                "load_error": self.load_error,
                "load_backup_path": (
                    str(self.load_backup_path) if self.load_backup_path else None
                ),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
