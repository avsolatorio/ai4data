"""Data adapters for loading data dictionaries into the canonical variable format.

The canonical format is a list of ``DictionaryVariable`` objects with fields:
- variable_name: str  (machine-readable identifier)
- label: str          (human-readable label)
- description: str    (optional extended text / question text)
- value_labels: dict  (optional code -> label mapping for categoricals)

Use ``ConfigurableDictionaryAdapter`` for CSV/JSON with custom column names,
or ``NADACatalogAdapter`` for NADA microdata catalog variable format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from .schemas import DictionaryVariable


# ----- Column name constants ----- #

REQUIRED_CANONICAL_COLUMNS = ["variable_name", "label"]
OPTIONAL_CANONICAL_COLUMNS = ["description", "value_labels"]

# Default CSV column mapping: canonical name -> source column name
CSV_DEFAULT_COLUMN_MAPPING: Dict[str, str] = {
    "variable_name": "variable_name",
    "label": "label",
    "description": "description",
    "value_labels": "value_labels",
}


# ----- Protocol ----- #


@runtime_checkable
class DictionaryAdapter(Protocol):
    """Protocol for data dictionary adapters."""

    def load_csv(self, path: str | Path, **kwargs: Any) -> List[DictionaryVariable]:
        """Load variables from a CSV file."""
        ...

    def load_json(self, path: str | Path) -> List[DictionaryVariable]:
        """Load variables from a JSON file (list of objects)."""
        ...

    def from_records(
        self, records: List[Dict[str, Any]]
    ) -> List[DictionaryVariable]:
        """Convert a list of dicts to DictionaryVariable objects."""
        ...


# ----- Helpers ----- #


def _build_rename_map(
    mapping: Dict[str, str], available_columns: List[str]
) -> Dict[str, str]:
    """Build source_col -> canonical_col rename map for columns that exist."""
    return {
        source: canonical
        for canonical, source in mapping.items()
        if source in available_columns
        and canonical in REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS
    }


def _row_to_variable(
    row: Dict[str, Any],
    *,
    validate_output: bool = True,
) -> Optional[DictionaryVariable]:
    """Convert a single dict row to a DictionaryVariable, skipping bad rows."""
    try:
        # value_labels may be a JSON string in CSV cells
        vl = row.get("value_labels")
        if isinstance(vl, str):
            try:
                vl = json.loads(vl)
            except (json.JSONDecodeError, ValueError):
                vl = None
        kwargs: Dict[str, Any] = {
            "variable_name": str(row["variable_name"]).strip(),
            "label": str(row["label"]).strip(),
        }
        if row.get("description"):
            kwargs["description"] = str(row["description"]).strip() or None
        if isinstance(vl, dict):
            kwargs["value_labels"] = {str(k): str(v) for k, v in vl.items()}
        return DictionaryVariable(**kwargs)
    except Exception:
        if validate_output:
            raise
        return None


# ----- Main adapter ----- #


class ConfigurableDictionaryAdapter:
    """Adapter that loads data dictionaries into canonical DictionaryVariable format.

    Supports CSV and JSON sources with configurable column name mapping.

    Parameters
    ----------
    mapping : dict, optional
        Maps canonical column names to source column names.
        Keys: ``variable_name``, ``label``, ``description``, ``value_labels``.
        Defaults to ``CSV_DEFAULT_COLUMN_MAPPING``.
    validate_output : bool
        If True, raise on rows missing required fields. If False, skip bad rows.

    Examples
    --------
    Default usage (CSV with "variable_name" and "label" columns):

    >>> adapter = ConfigurableDictionaryAdapter()
    >>> variables = adapter.load_csv("dictionary.csv")

    Custom column names:

    >>> adapter = ConfigurableDictionaryAdapter({"variable_name": "name", "label": "labl"})
    >>> variables = adapter.load_csv("nada_vars.csv")
    """

    def __init__(
        self,
        mapping: Optional[Dict[str, str]] = None,
        *,
        validate_output: bool = True,
    ):
        self.mapping = mapping or CSV_DEFAULT_COLUMN_MAPPING.copy()
        self.validate_output = validate_output

    def load_csv(self, path: str | Path, **kwargs: Any) -> List[DictionaryVariable]:
        """Load variables from a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.
        **kwargs
            Extra keyword arguments forwarded to ``pd.read_csv``.

        Returns
        -------
        list of DictionaryVariable
        """
        df = pd.read_csv(path, **kwargs)
        rename = _build_rename_map(self.mapping, list(df.columns))
        df = df.rename(columns=rename)
        # Keep only canonical columns that are present
        keep = [c for c in REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS
                if c in df.columns]
        df = df[keep]
        return self.from_records(df.to_dict("records"))

    def load_json(self, path: str | Path) -> List[DictionaryVariable]:
        """Load variables from a JSON file (array of objects).

        Parameters
        ----------
        path : str or Path
            Path to the JSON file.

        Returns
        -------
        list of DictionaryVariable
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # Accept {"variables": [...]} wrapper
            data = data.get("variables", list(data.values())[0])
        return self.from_records(data)

    def from_records(
        self, records: List[Dict[str, Any]]
    ) -> List[DictionaryVariable]:
        """Convert a list of dicts to DictionaryVariable objects.

        Parameters
        ----------
        records : list of dict
            Each dict should have at minimum ``variable_name`` and ``label`` keys
            (or the source column names configured in the mapping).

        Returns
        -------
        list of DictionaryVariable
        """
        # Apply mapping to each record
        def _remap(r: Dict[str, Any]) -> Dict[str, Any]:
            remapped: Dict[str, Any] = {}
            for canonical, source in self.mapping.items():
                if source in r:
                    remapped[canonical] = r[source]
                elif canonical in r:
                    remapped[canonical] = r[canonical]
            # Pass through any canonical columns already present
            for col in REQUIRED_CANONICAL_COLUMNS + OPTIONAL_CANONICAL_COLUMNS:
                if col not in remapped and col in r:
                    remapped[col] = r[col]
            return remapped

        results = []
        for rec in records:
            var = _row_to_variable(_remap(rec), validate_output=self.validate_output)
            if var is not None:
                results.append(var)
        return results


# ----- NADA Catalog Adapter ----- #


class NADACatalogAdapter:
    """Adapter for NADA microdata catalog variable metadata format.

    NADA variable objects use a nested JSON schema with fields:
    - ``name`` or ``nvar``: variable identifier
    - ``labl``: variable label
    - ``qstn.qstnlit``: question text (mapped to description)
    - ``catgry``: list of ``{catValu, labl}`` dicts (mapped to value_labels)

    Parameters
    ----------
    validate_output : bool
        If True, raise on variables missing required fields.

    Examples
    --------
    >>> adapter = NADACatalogAdapter()
    >>> variables = adapter.load_json("nada_catalog.json")
    """

    def __init__(self, *, validate_output: bool = True):
        self.validate_output = validate_output

    def load_json(self, path: str | Path) -> List[DictionaryVariable]:
        """Load variables from a NADA catalog JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the NADA catalog JSON file.

        Returns
        -------
        list of DictionaryVariable
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # NADA catalogs may wrap variables under different keys
        variables_raw = (
            data.get("variables")
            or data.get("var")
            or data.get("data_files", [{}])[0].get("variables", [])
            or []
        )
        return self.from_nada_variables(variables_raw)

    def from_nada_variables(
        self, nada_variables: List[Dict[str, Any]]
    ) -> List[DictionaryVariable]:
        """Convert a list of NADA variable dicts to DictionaryVariable objects.

        Parameters
        ----------
        nada_variables : list of dict
            Raw NADA variable objects.

        Returns
        -------
        list of DictionaryVariable
        """
        results = []
        for raw in nada_variables:
            var = self._parse_nada_variable(raw)
            if var is not None:
                results.append(var)
        return results

    def _parse_nada_variable(
        self, raw: Dict[str, Any]
    ) -> Optional[DictionaryVariable]:
        """Parse a single NADA variable dict."""
        try:
            # Variable name
            variable_name = (
                raw.get("name") or raw.get("nvar") or raw.get("id", "")
            )
            variable_name = str(variable_name).strip()

            # Label
            label = str(raw.get("labl") or raw.get("label") or "").strip()
            if not label:
                label = variable_name  # fall back to name if no label

            # Description from question text
            qstn = raw.get("qstn") or {}
            description = str(
                qstn.get("qstnlit") or raw.get("txt") or raw.get("description") or ""
            ).strip() or None

            # Value labels from catgry list
            catgry = raw.get("catgry") or raw.get("categories") or []
            value_labels: Optional[Dict[str, str]] = None
            if catgry:
                value_labels = {}
                for cat in catgry:
                    code = str(cat.get("catValu") or cat.get("code") or "").strip()
                    lbl = str(cat.get("labl") or cat.get("label") or "").strip()
                    if code:
                        value_labels[code] = lbl
                if not value_labels:
                    value_labels = None

            return DictionaryVariable(
                variable_name=variable_name,
                label=label,
                description=description,
                value_labels=value_labels,
            )
        except Exception:
            if self.validate_output:
                raise
            return None


# ----- Factory function ----- #


def adapter_from_config(
    mapping: Dict[str, str],
    *,
    validate_output: bool = True,
) -> ConfigurableDictionaryAdapter:
    """Create a configured dictionary adapter from a column mapping.

    Parameters
    ----------
    mapping : dict
        Maps canonical column names to source column names.
        Required keys: ``variable_name``, ``label``.
        Optional keys: ``description``, ``value_labels``.
    validate_output : bool
        If True, raise on rows missing required fields.

    Returns
    -------
    ConfigurableDictionaryAdapter
    """
    return ConfigurableDictionaryAdapter(mapping, validate_output=validate_output)
