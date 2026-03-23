"""Feedback system for anomaly explanation reviewers.

Schema and storage for reviewer feedback on LLM-generated explanations.
Uses (indicator_code, geography_code, window_str) as stable key for lookups.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Feedback schema for app integration
FEEDBACK_SCHEMA = {
    "item_id": "int (index into review items)",
    "indicator_code": "str",
    "geography_code": "str",
    "window_str": "str",
    "verdict": "approved | rejected | needs_review",
    "comment": "str (optional)",
    "suggested_classification": "str (optional, if reviewer disagrees)",
    "timestamp": "ISO8601",
}

DEFAULT_FEEDBACK_FILENAME = "anomaly_feedback.json"

# In-memory store (persist to JSON file for production)
_feedback_store: List[Dict[str, Any]] = []
_feedback_path: Optional[Path] = None


def _feedback_key(entry: Dict[str, Any]) -> tuple:
    """Stable key for matching feedback to an anomaly item."""
    return (
        entry.get("indicator_code", ""),
        entry.get("geography_code", ""),
        entry.get("window_str", ""),
    )


def _persist_store() -> None:
    """Write feedback store to file if path is set."""
    global _feedback_path, _feedback_store
    if _feedback_path:
        import json

        _feedback_path.parent.mkdir(parents=True, exist_ok=True)
        _feedback_path.write_text(
            json.dumps(_feedback_store, indent=2, default=str),
            encoding="utf-8",
        )


def init_feedback_store(path: str | Path | None = None) -> None:
    """Initialize feedback storage, loading from file if it exists.

    If path is None, uses DEFAULT_FEEDBACK_FILENAME in current working directory,
    so feedback is always persisted by default.
    """
    global _feedback_store, _feedback_path
    _feedback_path = Path(path) if path is not None else Path.cwd() / DEFAULT_FEEDBACK_FILENAME
    _feedback_store = []
    if _feedback_path.exists():
        import json

        try:
            _feedback_store = json.loads(_feedback_path.read_text())
            if not isinstance(_feedback_store, list):
                _feedback_store = []
        except Exception:
            _feedback_store = []


def submit_feedback(
    item_id: int,
    indicator_code: str,
    geography_code: str,
    window_str: str,
    verdict: str,
    comment: Optional[str] = None,
    suggested_classification: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit reviewer feedback for an anomaly item.

    Uses upsert by (indicator_code, geography_code, window_str): if feedback
    already exists for this anomaly, it is updated; otherwise a new entry is appended.
    """
    if verdict not in ("approved", "rejected", "needs_review"):
        raise ValueError("verdict must be approved, rejected, or needs_review")

    key = (indicator_code, geography_code, window_str)
    entry = {
        "item_id": item_id,
        "indicator_code": indicator_code,
        "geography_code": geography_code,
        "window_str": window_str,
        "verdict": verdict,
        "comment": comment or "",
        "suggested_classification": suggested_classification or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    for i, existing in enumerate(_feedback_store):
        if _feedback_key(existing) == key:
            _feedback_store[i] = entry
            _persist_store()
            return entry

    _feedback_store.append(entry)
    _persist_store()
    return entry


def get_feedback(
    item_id: Optional[int] = None,
    indicator_code: Optional[str] = None,
    geography_code: Optional[str] = None,
    window_str: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve feedback, optionally filtered."""
    out = _feedback_store
    if item_id is not None:
        out = [e for e in out if e["item_id"] == item_id]
    if indicator_code is not None:
        out = [e for e in out if e["indicator_code"] == indicator_code]
    if geography_code is not None:
        out = [e for e in out if e["geography_code"] == geography_code]
    if window_str is not None:
        out = [e for e in out if e["window_str"] == window_str]
    return out


def get_feedback_for_item(
    indicator_code: str,
    geography_code: str,
    window_str: str,
) -> Optional[Dict[str, Any]]:
    """Get the most recent feedback for an anomaly item by stable key."""
    matches = get_feedback(
        indicator_code=indicator_code,
        geography_code=geography_code,
        window_str=window_str,
    )
    return matches[-1] if matches else None


def export_feedback_csv(path: str | Path) -> Path:
    """Export all feedback to CSV for downstream use."""
    import pandas as pd

    path = Path(path)
    df = pd.DataFrame(_feedback_store)
    df.to_csv(path, index=False)
    return path
