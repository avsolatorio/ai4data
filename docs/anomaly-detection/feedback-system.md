# Reviewer Feedback System

The Anomaly Explanation Reviewer app supports collecting structured feedback from reviewers. This document describes the system and how to integrate it.

## Schema

Each feedback entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | int | Index into the review items list |
| `indicator_code` | str | Indicator code (e.g., WB_CSC_SI_POV_UMIC) |
| `geography_code` | str | Geography code (e.g., ISL) |
| `window_str` | str | Anomaly window as "start-end" |
| `verdict` | str | `approved`, `rejected`, or `needs_review` |
| `comment` | str (optional) | Free-text reviewer comment |
| `suggested_classification` | str (optional) | If reviewer disagrees with LLM classification |
| `timestamp` | ISO8601 | When feedback was submitted |

## API Endpoints

- **POST /api/feedback** — Submit feedback (JSON body)
- **GET /api/feedback** — List feedback (optional `?item_id=N`)
- **GET /api/feedback/schema** — Get schema for integration
- **GET /api/feedback/export** — Export all feedback to CSV

## Usage

### Running with persistent feedback

```bash
uv run python -m apps.anomaly_review path/to/review.json feedback.json
```

Feedback is appended to `feedback.json` on each submission.

### Exporting feedback

```bash
curl http://localhost:8000/api/feedback/export
```

Or use the Python API:

```python
from apps.anomaly_review.feedback import export_feedback_csv, init_feedback_store

init_feedback_store("feedback.json")
export_feedback_csv("feedback_export.csv")
```

### Downstream use

The CSV export can be used to:

1. Retrain or fine-tune models
2. Audit review coverage
3. Compute inter-rater agreement
4. Update classification labels in the source system
