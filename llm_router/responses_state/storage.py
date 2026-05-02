"""Storage boundary for the Responses state machine.

The durable implementation currently reuses ``llm_router.session_store``. This
module exists so provider adapters and HTTP handlers do not grow direct
dependencies on the storage layout as the state machine evolves.
"""

from llm_router.session_store import ResponsesSession, SessionStore

__all__ = ["ResponsesSession", "SessionStore"]
