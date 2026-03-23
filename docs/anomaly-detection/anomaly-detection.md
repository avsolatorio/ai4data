# Anomaly Detection in Development Data

This section covers tools for detecting and explaining anomalies in timeseries indicator data using LLMs.

## Contents

- [Timeseries Anomaly Explanation with LLMs](../../notebooks/data-anomaly/Timeseries_Anomaly_Explanation_with_LLMs.ipynb) — Step-by-step notebook
- [Reviewer Feedback System](feedback-system.md) — Collecting reviewer feedback on explanations

## Quick Start

```python
from ai4data.anomaly.explanation import (
    ScorecardWideAdapter,
    extract_anomaly_contexts,
    parse_batch_output,
    export_for_review,
)
```

See the notebook for the full pipeline.