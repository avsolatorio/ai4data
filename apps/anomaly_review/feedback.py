"""Feedback system for anomaly explanation reviewers.

Schema and storage for reviewer feedback on LLM-generated explanations.
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

# In-memory store (persist to JSON file for production)
_feedback_store: List[Dict[str, Any]] = []
_feedback_path: Optional[Path] = None


def init_feedback_store(path: str | Path | None = None) -> None:
    """Initialize feedback storage, optionally loading from file."""
    global _feedback_store, _feedback_path
    _feedback_path = Path(path) if path else None
    _feedback_store = []
    if _feedback_path and _feedback_path.exists():
        import json

        _feedback_store = json.loads(_feedback_path.read_text())


def submit_feedback(
    item_id: int,
    indicator_code: str,
    geography_code: str,
    window_str: str,
    verdict: str,
    comment: Optional[str] = None,
    suggested_classification: Optional[str] = None,
) -> Dict[str, Any]:
    """Submit reviewer feedback for an anomaly item."""
    if verdict not in ("approved", "rejected", "needs_review"):
        raise ValueError("verdict must be approved, rejected, or needs_review")

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
    _feedback_store.append(entry)

    if _feedback_path:
        import json

        _feedback_path.write_text(
            json.dumps(_feedback_store, indent=2, default=str),
            encoding="utf-8",
        )

    return entry


def get_feedback(
    item_id: Optional[int] = None,
    indicator_code: Optional[str] = None,
    geography_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve feedback, optionally filtered."""
    out = _feedback_store
    if item_id is not None:
        out = [e for e in out if e["item_id"] == item_id]
    if indicator_code is not None:
        out = [e for e in out if e["indicator_code"] == indicator_code]
    if geography_code is not None:
        out = [e for e in out if e["geography_code"] == geography_code]
    return out


def export_feedback_csv(path: str | Path) -> Path:
    """Export all feedback to CSV for downstream use."""
    import pandas as pd

    path = Path(path)
    df = pd.DataFrame(_feedback_store)
    df.to_csv(path, index=False)
    return path
