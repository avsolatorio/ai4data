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
from ai4data.anomaly_detection.adapters import (
    CANONICAL_COLUMNS,
    ConfigurableAdapter,
    ScorecardWideAdapter,
    adapter_from_config,
)
from ai4data.anomaly_detection.context import extract_anomaly_contexts
from ai4data.anomaly_detection.arbiter import (
    build_arbiter_payload,
    group_explanations_by_context_with_providers,
    harmonize_explanations,
)
from ai4data.anomaly_detection.explainers import list_explainers, register_explainer
from ai4data.anomaly_detection.batch_builder import build_batch_file, list_batch_providers
from ai4data.anomaly_detection.batch_runner import (
    download_batch_output,
    run_batch,
    submit_batch,
    wait_for_batch,
)
from ai4data.anomaly_detection.output_parser import parse_batch_output
from ai4data.anomaly_detection.review_output import (
    export_for_review,
    export_for_review_with_explainers,
    to_review_format,
    to_review_format_with_explainers,
)
from ai4data.anomaly_detection.schemas import (
    Anomaly,
    AnomalyExplanation,
    Classification,
    EvidenceSource,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "ConfigurableAdapter",
    "ScorecardWideAdapter",
    "adapter_from_config",
    "extract_anomaly_contexts",
    "build_batch_file",
    "list_batch_providers",
    "submit_batch",
    "wait_for_batch",
    "download_batch_output",
    "run_batch",
    "parse_batch_output",
    "to_review_format",
    "to_review_format_with_explainers",
    "export_for_review",
    "export_for_review_with_explainers",
    "build_arbiter_payload",
    "group_explanations_by_context_with_providers",
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
