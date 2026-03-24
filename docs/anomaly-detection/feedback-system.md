# Reviewer Feedback System

The Anomaly Explanation Reviewer application supports collecting structured feedback from domain experts and data stewards. This document describes the feedback system, how to run the application, and how to use collected feedback downstream.

---

## Why Reviewer Feedback Matters

LLM-elicited anomaly explanations are hypotheses, not facts. Even with schema enforcement and conservative prompting, models can propose incorrect classifications, cite the wrong events, or miss domain-specific knowledge that only an expert would know. The feedback system closes the human-in-the-loop by creating a structured mechanism for reviewers to:

1. **Validate** LLM outputs that are correct (approved verdicts build confidence in the pipeline)
2. **Correct** wrong classifications or evidence (rejected + suggested_classification improves coverage)
3. **Surface uncertainty** by flagging explanations that need further investigation before a decision is made
4. **Build a labeled dataset** of reviewed anomalies that can be used for model improvement, inter-rater agreement analysis, or audit

Without a feedback mechanism, AI-assisted quality assurance is a one-way system—outputs flow in only one direction and there is no way to measure accuracy, track improvements, or detect systematic errors.

---

## Feedback Schema

Each feedback entry records:

| Field | Type | Description |
|---|---|---|
| `item_id` | int | Index into the review items list |
| `indicator_code` | str | Indicator code (e.g., `NY.GDP.MKTP.KD.ZG`) |
| `geography_code` | str | ISO 3166-1 alpha-3 code (e.g., `NGA`) |
| `window_str` | str | Anomaly window as `"start-end"` (e.g., `"2015-2016"`) |
| `verdict` | str | `approved`, `rejected`, or `needs_review` |
| `comment` | str (optional) | Free-text reviewer comment |
| `suggested_classification` | str (optional) | Alternative classification if reviewer disagrees |
| `timestamp` | ISO8601 | When feedback was submitted |

The combination of (`indicator_code`, `geography_code`, `window_str`) forms the **stable key** for a feedback entry. Resubmitting feedback for the same stable key updates the existing entry (upsert), so reviewers can revise their verdicts without creating duplicates.

---

## Running the Review Application

The review application is a FastAPI server with a single-page UI. It loads a review JSON payload and stores feedback to disk.

```bash
# Start with a review file and a feedback persistence file
uv run python -m apps.anomaly_review path/to/review.json feedback.json

# Navigate to http://localhost:8000
```

The `review.json` file is produced by `export_for_review()` or `export_for_review_with_explainers()` from the explanation pipeline:

```python
from ai4data.anomaly.explanation import export_for_review

export_for_review(explanations, output_path="review.json")
```

The review UI displays:
- A navigation list of all anomaly items (indicator + country + window)
- A timeseries chart for each item, highlighting the anomaly window
- The LLM-generated classification, confidence, explanation, and evidence sources
- (When multiple explainers are used) each explainer's output alongside the judge verdict
- Feedback controls: verdict buttons, free-text comment, optional suggested classification

---

## API Endpoints

The review app exposes a REST API for programmatic integration:

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/review` | Full review payload (all items) |
| `GET` | `/api/items` | Navigation list (id, indicator, geography, window) |
| `GET` | `/api/items/{item_id}` | Single item with full explanation details |
| `POST` | `/api/feedback` | Submit feedback (JSON body matching schema) |
| `GET` | `/api/feedback` | List all feedback (optional `?item_id=N` filter) |
| `GET` | `/api/feedback/item` | Get feedback by stable key (`indicator_code`, `geography_code`, `window_str`) |
| `GET` | `/api/feedback/schema` | Feedback schema for integration |
| `GET` | `/api/feedback/export` | Export all feedback as CSV |

---

## Exporting Feedback

Feedback can be exported at any time during or after the review session:

```bash
# Via HTTP
curl http://localhost:8000/api/feedback/export > feedback_export.csv

# Or via the Python API (when running programmatically)
from apps.anomaly_review.feedback import export_feedback_csv, init_feedback_store

init_feedback_store("feedback.json")
export_feedback_csv("feedback_export.csv")
```

The CSV export includes all schema fields plus a derived `stable_key` column for joining with the review data.

---

## Analyzing Collected Feedback

The CSV export supports several downstream analyses:

### Coverage analysis

```python
import pandas as pd

fb = pd.read_csv("feedback_export.csv")
total = len(fb)
approved = (fb["verdict"] == "approved").sum()
rejected = (fb["verdict"] == "rejected").sum()
needs_review = (fb["verdict"] == "needs_review").sum()

print(f"Total reviewed: {total}")
print(f"Approval rate:  {approved/total:.1%}")
print(f"Rejection rate: {rejected/total:.1%}")
```

### Classification accuracy

```python
# Compare LLM classification to reviewer's suggested classification
corrections = fb[fb["suggested_classification"].notna()].copy()
corrections["agreed"] = (
    corrections["verdict"] == "approved"
)
print(corrections[["indicator_code", "geography_code", "window_str",
                    "verdict", "suggested_classification"]].head())
```

### Inter-rater agreement (when multiple reviewers)

```python
# If multiple reviewers submit feedback for the same item,
# compute Cohen's kappa over verdict values
from sklearn.metrics import cohen_kappa_score

# Join on stable_key where reviewer_id differs
# ... (join logic depends on your data collection setup)
```

### Feeding back into the pipeline

Reviewed and corrected labels can be used to:
1. **Evaluate** pipeline quality by computing precision/recall of LLM classifications against expert verdicts
2. **Retrain or fine-tune** a downstream classifier using `(context, classification)` pairs
3. **Refine prompts** by identifying systematic misclassifications (e.g., the model consistently confuses `measurement_system_update` with `external_driver` for particular indicator types)
4. **Update classification labels** in the source system where the LLM explanation was correct and the original label was missing or wrong

---

## Implementation Reference

The feedback system implementation is in [`apps/anomaly_review/feedback.py`](../../../apps/anomaly_review/feedback.py) and [`apps/anomaly_review/main.py`](../../../apps/anomaly_review/main.py).
