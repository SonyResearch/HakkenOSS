from typing import Any

import numpy as np
import pandas as pd

from hakken_models.results.entities import MetricInfo, ModelInfo


def format_metric_value(x: Any, decimals: int) -> str:
    """Format numeric metrics with fixed decimals; NaNs as '-'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{float(x):.{decimals}f}"


def latex_bold(s: str) -> str:
    """Wrap a LaTeX-safe numeric string in math bold."""
    if s in ("", "-"):
        return s
    return rf"$\mathbf{{{s}}}$"


def sort_rows_by_model_position(
    df: pd.DataFrame,
    model_name_mapping: dict[str, ModelInfo],
    model_col: str = "model",
) -> pd.DataFrame:
    """Sort rows based on ModelInfo.position (before model name mapping)."""
    if model_col not in df.columns:
        return df

    out = df.copy()
    out["_row_order"] = out[model_col].map(
        lambda m: model_name_mapping.get(m, ModelInfo(str(m), 999, "#7f7f7f")).position
    )
    return out.sort_values("_row_order", ascending=True).drop(columns=["_row_order"])


def map_model_names(
    df: pd.DataFrame,
    model_name_mapping: dict[str, ModelInfo],
    model_col: str = "model",
) -> pd.DataFrame:
    """Replace model identifiers with ModelInfo.display_name."""
    if model_col not in df.columns:
        return df

    out = df.copy()
    out[model_col] = out[model_col].map(
        lambda m: model_name_mapping.get(m, ModelInfo(str(m), 999, "#7f7f7f")).display_name
    )
    return out


def format_metric_column_for_latex(
    series: pd.Series,
    metric_info: MetricInfo,
    decimals: int = 3,
) -> pd.Series:
    """
    Format a single metric series:
    - numeric formatting
    - '-' for all-NaN columns
    - bold the best value (max if higher_is_better else min)
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # If all NaN: fill with '-'
    if numeric.notna().sum() == 0:
        return pd.Series(["-"] * len(series), index=series.index)

    best_val = numeric.max() if metric_info.higher_is_better else numeric.min()
    best_mask = numeric.eq(best_val) & numeric.notna()

    formatted = numeric.map(lambda v: format_metric_value(v, decimals))
    return formatted.where(~best_mask, formatted.map(latex_bold))


def format_metrics_for_latex(
    df: pd.DataFrame,
    metric_mapping: dict[str, MetricInfo],
    decimals: int = 3,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """Apply LaTeX formatting + best-value bolding to all metric columns."""
    out = df.copy()
    for metric_col, metric_info in metric_mapping.items():
        if metric_col not in out.columns:
            continue
        out[metric_col] = format_metric_column_for_latex(
            out[metric_col] * multiplier,
            metric_info=metric_info,
            decimals=decimals,
        )
    return out


def rename_columns_for_latex(
    df: pd.DataFrame,
    rename_cols: dict[str, str] | None,
) -> pd.DataFrame:
    """Rename columns to LaTeX-friendly names."""
    if rename_cols is None:
        return df
    return df.rename(columns=rename_cols)


def format_df_for_latex(
    df: pd.DataFrame,
    metric_mapping: dict[str, MetricInfo],
    decimals: int = 3,
    multiplier: float = 1.0,
    rename_cols: dict[str, str] | None = None,
    model_name_mapping: dict[str, ModelInfo] | None = None,
    sort_rows: bool = True,
    format_metrics: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    # ---- Row ordering (before model renaming!)
    if sort_rows and model_name_mapping is not None:
        out = sort_rows_by_model_position(out, model_name_mapping)

    # ---- Optional: map model names (after sorting)
    if model_name_mapping is not None:
        out = map_model_names(out, model_name_mapping)

    # ---- Format metric columns + bold best
    if format_metrics:
        out = format_metrics_for_latex(
            out, metric_mapping, decimals=decimals, multiplier=multiplier
        )

    # ---- Optional: rename columns to LaTeX names
    return rename_columns_for_latex(out, rename_cols)
