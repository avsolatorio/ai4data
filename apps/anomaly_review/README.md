# Anomaly Explanation Reviewer App

A FastAPI + static frontend app for reviewing LLM-generated anomaly explanations alongside timeseries charts.

## Run

```bash
# From repo root — loads sample data by default (7 example anomalies)
uv run python -m apps.anomaly_review

# Or with your own anomaly review JSON
uv run python -m apps.anomaly_review path/to/anomaly_review.json
```

Open http://localhost:8000

**Features:** Dark/light mode toggle, anomaly window highlighting on charts, diverse example classifications.

Or use the pipeline to export review format first:

```python
from ai4data.anomaly.explanation import export_for_review, parse_batch_output

anomalies_df = parse_batch_output("output.jsonl", provider="openai", ...)
export_for_review(anomalies_df, timeseries_df=canonical_df, output_path="anomaly_review.json")
```

Then:
```bash
uv run python -m apps.anomaly_review.main anomaly_review.json
```

Open http://localhost:8000

## API

- `GET /` — Reviewer UI
- `GET /api/review` — Full review payload
- `GET /api/items` — List of anomalies for navigation
- `GET /api/items/{id}` — Single anomaly with timeseries
