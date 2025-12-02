"""
Shared store management for cost tracking across agent graphs.
"""

from typing import Optional

from langgraph.store.memory import InMemoryStore

from src.cost_calc.store_cost_tracker import StoreCostTracker

# Global shared store instance for cost tracking
_shared_store = None
_current_session_tracker = None


def get_shared_store() -> InMemoryStore:
    """Get or create the shared store instance."""
    global _shared_store
    if _shared_store is None:
        _shared_store = InMemoryStore()
    return _shared_store


def get_or_create_cost_tracker(session_id: Optional[str] = None) -> StoreCostTracker:
    """Get or create a cost tracker for the current session."""
    global _current_session_tracker
    store = get_shared_store()

    if _current_session_tracker is None or (
        _current_session_tracker.session_id != session_id if session_id else False
    ):
        _current_session_tracker = StoreCostTracker(store, session_id)

    return _current_session_tracker


def finalize_current_session():
    """Finalize the current cost tracking session."""
    global _current_session_tracker
    if _current_session_tracker:
        _current_session_tracker.finalize_session()
        _current_session_tracker = None


def reset_shared_store():
    """Reset the shared store (useful for testing)."""
    global _shared_store, _current_session_tracker
    _shared_store = None
    _current_session_tracker = None
