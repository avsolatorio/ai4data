"""ai4data.anomaly_detection - Anomaly detection and explanation in timeseries data.

This module provides tools for detecting anomalies in timeseries indicators
and generating LLM-based explanations. It requires optional dependencies:

    uv pip install ai4data[anomaly]

Example usage:
    from ai4data.anomaly_detection import (
        ScorecardWideAdapter,
        extract_anomaly_contexts,
        parse_batch_output,
    )
    from ai4data.anomaly_detection.prompts import (
        SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE,
        get_anomaly_response_format,
    )

    adapter = ScorecardWideAdapter()
    df = adapter.load("wide.csv", "anomalies.csv")
"""

__version__ = "0.1.0"

from ai4data.anomaly_detection import (
    adapters,
    arbiter,
    context,
    explainers,
    output_parser,
    prompts,
    schemas,
)
from ai4data.anomaly_detection.adapters import ScorecardWideAdapter, adapter_from_config
from ai4data.anomaly_detection.context import extract_anomaly_contexts
from ai4data.anomaly_detection.arbiter import (
    build_arbiter_payload,
    harmonize_explanations,
)
from ai4data.anomaly_detection.explainers import list_explainers, register_explainer
from ai4data.anomaly_detection.output_parser import parse_batch_output
from ai4data.anomaly_detection.review_output import export_for_review, to_review_format
from ai4data.anomaly_detection.schemas import (
    Anomaly,
    AnomalyExplanation,
    Classification,
    EvidenceSource,
)

__all__ = [
    "ScorecardWideAdapter",
    "adapter_from_config",
    "extract_anomaly_contexts",
    "parse_batch_output",
    "to_review_format",
    "export_for_review",
    "build_arbiter_payload",
    "harmonize_explanations",
    "register_explainer",
    "list_explainers",
    "Anomaly",
    "AnomalyExplanation",
    "Classification",
    "EvidenceSource",
    "adapters",
    "arbiter",
    "context",
    "explainers",
    "output_parser",
    "prompts",
    "schemas",
]
