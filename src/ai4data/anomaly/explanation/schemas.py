"""Pydantic schemas for anomaly detection and explanation.

This module defines the canonical data structures used throughout the
timeseries anomaly explanation pipeline.
"""

from enum import Enum
from typing import Annotated, List

from pydantic import BaseModel, ConfigDict, Field


# ----- Canonical input schema (for pipeline consumption) ----- #

# Canonical DataFrame columns (referenced in adapters and context):
# - indicator_id: str
# - indicator_name: str
# - geography_id: str
# - geography_name: str
# - period: int  (year for annual data)
# - value: float
# - is_imputed: bool
# - anomaly_score: float  (e.g., |z-score|)
# - outlier_count: int
# - freq: str (optional, e.g., "A" for annual)


# ----- LLM response schema ----- #


class Classification(str, Enum):
    """Primary classification of an anomaly's cause."""

    data_error = "data_error"
    external_driver = "external_driver"
    measurement_system_update = "measurement_system_update"
    modeling_artifact = "modeling_artifact"
    insufficient_data = "insufficient_data"


class EvidenceStrength(str, Enum):
    """Strength of evidence supporting the explanation."""

    strong_direct = "strong_direct"
    moderate_contextual = "moderate_contextual"
    weak_speculative = "weak_speculative"
    no_evidence = "no_evidence"


class EvidenceSourceType(str, Enum):
    """Type of evidence source."""

    global_event = "global_event"
    regional_event = "regional_event"
    policy_change = "policy_change"
    data_revision = "data_revision"
    rebasing = "rebasing"
    natural_disaster = "natural_disaster"
    conflict = "conflict"
    economic_crisis = "economic_crisis"
    data_error = "data_error"
    other = "other"


class Verifiability(str, Enum):
    """Verifiability level of the evidence."""

    well_documented = "well_documented"
    partially_documented = "partially_documented"
    uncertain = "uncertain"
    not_applicable = "not_applicable"


class Source(str, Enum):
    """Source of the anomaly explanation."""

    llm_inferred = "llm_inferred"


class StrictBaseModel(BaseModel):
    """Base model with strict validation (no extra fields)."""

    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceSource(StrictBaseModel):
    """A cited evidence source for an anomaly explanation."""

    name: str
    date_range: str
    source_type: EvidenceSourceType
    verifiability: Verifiability


class Anomaly(StrictBaseModel):
    """Single anomaly explanation from the LLM."""

    window: Annotated[
        List[int],
        Field(
            min_length=2,
            max_length=2,
            description="Inclusive [start_year, end_year] of the anomaly window.",
        ),
    ]
    is_anomaly: bool
    classification: Classification
    confidence: Annotated[float, Field(ge=0, le=1)]
    explanation: str
    evidence_strength: EvidenceStrength
    evidence_source: List[EvidenceSource]
    source: Source


class AnomalyExplanation(StrictBaseModel):
    """Container for LLM-generated anomaly explanations."""

    anomalies: List[Anomaly]
