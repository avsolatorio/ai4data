"""Pydantic schemas for data dictionary augmentation.

This module defines the canonical data structures used throughout the
metadata augmentation pipeline: loading variables, generating themes,
and assembling the augmented dictionary output.
"""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ----- Base ----- #


class StrictBaseModel(BaseModel):
    """Base model with strict validation (no extra fields)."""

    model_config = ConfigDict(extra="forbid", strict=True)


# ----- Input schema ----- #


class DictionaryVariable(StrictBaseModel):
    """A single variable entry in a data dictionary.

    Parameters
    ----------
    variable_name : str
        The machine-readable variable identifier (e.g., "HV001", "age_hh_head").
    label : str
        Human-readable label describing the variable (e.g., "Cluster number").
    description : str, optional
        Extended description or question text for the variable.
    value_labels : dict, optional
        Mapping of code strings to label strings for categorical variables
        (e.g., {"1": "Male", "2": "Female"}).
    """

    variable_name: str
    label: str
    description: Optional[str] = None
    value_labels: Optional[Dict[str, str]] = None


# ----- LLM response schema ----- #


class ThemeGenerationResult(StrictBaseModel):
    """Raw LLM output for a single cluster's theme.

    This schema is used to validate the JSON response from the LLM at the
    API boundary. It is then converted to a ``Theme`` in the augmented output.

    Parameters
    ----------
    theme_name : str
        2–6 word title-case theme name (e.g., "Household Asset Ownership").
    description : str
        1–2 sentence description grounded in the variable labels.
    example_variables : list of str
        Up to 5 variable names from the cluster that best represent the theme.
    """

    theme_name: str
    description: str
    example_variables: List[str]


# ----- Output schema ----- #


class Theme(StrictBaseModel):
    """A validated, LLM-generated thematic cluster.

    Parameters
    ----------
    theme_name : str
        2–6 word title-case theme name.
    description : str
        1–2 sentence description of the theme.
    example_variables : list of str
        1–5 representative variable names from this theme.
    """

    theme_name: str
    description: str
    example_variables: Annotated[
        List[str],
        Field(
            min_length=1,
            max_length=5,
            description="Representative variable names from this theme.",
        ),
    ]


class ThemeAssignment(StrictBaseModel):
    """Maps a single variable to its assigned theme and cluster.

    Parameters
    ----------
    variable_name : str
        The variable identifier.
    theme_name : str
        The name of the theme this variable was assigned to.
    cluster_id : int
        The numeric cluster ID from the clustering step.
    """

    variable_name: str
    theme_name: str
    cluster_id: int


class AugmentedDictionary(StrictBaseModel):
    """Top-level output of the augmentation pipeline.

    Contains the generated themes and a full mapping of every variable to
    its theme, along with optional run metadata (model, timestamp, config).

    Parameters
    ----------
    dataset_id : str, optional
        Identifier for the source dataset (e.g., NADA survey ID).
    themes : list of Theme
        All generated themes, one per cluster.
    variable_assignments : list of ThemeAssignment
        One entry per input variable, mapping it to a theme and cluster.
    metadata : dict, optional
        Run configuration: model name, embedding model, timestamp, etc.
    """

    dataset_id: Optional[str] = None
    themes: List[Theme]
    variable_assignments: List[ThemeAssignment]
    metadata: Optional[Dict[str, Any]] = None
