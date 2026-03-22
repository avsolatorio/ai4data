"""Data adapters for converting legacy formats to canonical timeseries anomaly format.

The canonical format uses these column names:
- indicator_id, indicator_name
- geography_id, geography_name
- period, value, is_imputed
- anomaly_score, outlier_count
- freq (optional)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# Default Scorecard column mapping (wide format + anomaly scores)
SCORECARD_COLUMN_MAPPING = {
    "indicator_id": "INDICATOR",
    "indicator_name": "INDICATOR_LABEL",
    "geography_id": "REF_AREA",
    "geography_name": "REF_AREA_LABEL",
    "period": "YEAR",
    "value": "VALUE",
    "is_imputed": "Imputed",
    "anomaly_score": "absZscore",
    "outlier_count": "outlier_indicator_total",
    "freq": "FREQ",
}


def _detect_year_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that represent years (numeric)."""
    year_cols = []
    for col in df.columns:
        try:
            int(col)
            year_cols.append(col)
        except (ValueError, TypeError):
            pass
    return year_cols


# Canonical column names
CANONICAL_COLUMNS = [
    "indicator_id",
    "indicator_name",
    "geography_id",
    "geography_name",
    "period",
    "value",
    "is_imputed",
    "anomaly_score",
    "outlier_count",
    "freq",
]


def adapter_from_config(mapping: Dict[str, str]):
    """Create a configured adapter function from a column mapping.

    Parameters
    ----------
    mapping : dict
        Maps canonical column names to source column names.
        Required keys: indicator_id, indicator_name, geography_id, geography_name,
        period, value, is_imputed, anomaly_score, outlier_count.

    Returns
    -------
    callable
        An adapter function(wide_path, anomaly_path) -> pd.DataFrame.
    """

    def adapt(wide_path: str | Path, anomaly_path: str | Path) -> pd.DataFrame:
        wide_df = pd.read_csv(wide_path)
        raw_df = pd.read_csv(anomaly_path)

        # Add anomaly_score if we have Zscore
        if "Zscore" in raw_df.columns and "absZscore" not in raw_df.columns:
            raw_df["absZscore"] = raw_df["Zscore"].abs()

        year_cols = _detect_year_columns(wide_df)
        non_year_cols = [c for c in wide_df.columns if c not in year_cols]

        long_df = pd.melt(
            wide_df,
            id_vars=non_year_cols,
            value_vars=year_cols,
            var_name=mapping.get("period", "YEAR"),
            value_name=mapping.get("value", "VALUE"),
        )
        long_df[mapping["period"]] = pd.to_numeric(
            long_df[mapping["period"]], errors="coerce"
        ).astype("Int64")

        common_cols = [c for c in raw_df.columns if c in long_df.columns]
        o_df = long_df.merge(raw_df, on=common_cols, how="left")

        # Keep only (indicator, geography) pairs from anomaly file, expand to full series
        anomaly_keys = raw_df[
            [mapping["indicator_id"], mapping["geography_id"]]
        ].drop_duplicates()
        result = anomaly_keys.merge(
            o_df,
            on=[mapping["indicator_id"], mapping["geography_id"]],
            how="left",
        )

        # Ensure is_imputed is bool
        imputed_col = mapping.get("is_imputed", "Imputed")
        if imputed_col in result.columns:
            result[imputed_col] = result[imputed_col].fillna(False).astype(bool)

        # Rename to canonical (only cols that exist)
        reverse_mapping = {
            v: k for k, v in mapping.items() if v in result.columns and k in CANONICAL_COLUMNS
        }
        return result.rename(columns=reverse_mapping)

    return adapt


class ScorecardWideAdapter:
    """Adapter for World Bank Scorecard wide-format + anomaly scores CSVs."""

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """Initialize with optional custom column mapping.

        Parameters
        ----------
        column_mapping : dict, optional
            Override default Scorecard mapping. Keys are canonical names.
        """
        self.mapping = column_mapping or SCORECARD_COLUMN_MAPPING.copy()

    def load(
        self,
        wide_path: str | Path,
        anomaly_path: str | Path,
    ) -> pd.DataFrame:
        """Load and convert to canonical format.

        Parameters
        ----------
        wide_path : str or Path
            Path to wide-format CSV (metadata cols + year columns).
        anomaly_path : str or Path
            Path to anomaly scores CSV.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with canonical column names.
        """
        adapt = adapter_from_config(self.mapping)
        return adapt(wide_path, anomaly_path)
