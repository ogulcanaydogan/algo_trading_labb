"""
Data quality and leakage checks for ML training datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "label",
    "target",
}
TARGET_PREFIXES = ("target_", "future_")


def get_feature_columns(
    df: pd.DataFrame,
    extra_exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """Get safe feature columns excluding targets and future leaks."""
    exclude = set(DEFAULT_EXCLUDE_COLUMNS)
    if extra_exclude:
        exclude.update(extra_exclude)

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith(TARGET_PREFIXES):
            continue
        feature_cols.append(col)
    return feature_cols


def validate_feature_leakage(feature_cols: Iterable[str]) -> List[str]:
    """Return any columns that look like leakage targets or future data."""
    leakage = []
    for col in feature_cols:
        if col.startswith(TARGET_PREFIXES) or col in {"label", "target"}:
            leakage.append(col)
    return leakage


def validate_target_alignment(
    df: pd.DataFrame,
    price_col: str = "close",
    target_col: str = "target_return",
    horizon: int = 1,
    min_corr: float = 0.9,
) -> List[str]:
    """Validate target alignment by comparing to expected forward return."""
    warnings: List[str] = []
    if target_col not in df.columns or price_col not in df.columns:
        return warnings

    expected = df[price_col].pct_change(horizon).shift(-horizon)
    aligned = pd.concat([expected, df[target_col]], axis=1).dropna()
    if aligned.empty:
        return warnings

    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if corr is None or np.isnan(corr):
        warnings.append(f"Target alignment check failed: correlation undefined for {target_col}.")
        return warnings

    if corr < min_corr:
        warnings.append(f"Target alignment low: corr={corr:.2f} for {target_col} (min {min_corr}).")
    return warnings


def _robust_outlier_rate(series: pd.Series, threshold: float = 4.0) -> float:
    """Compute outlier rate using robust z-score (median/MAD)."""
    values = series.dropna().values
    if values.size < 10:
        return 0.0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return 0.0
    robust_z = 0.6745 * (values - median) / mad
    return float(np.mean(np.abs(robust_z) > threshold))


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between baseline and recent samples."""
    if expected.size < 20 or actual.size < 20:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / max(1, expected_counts.sum())
    actual_perc = actual_counts / max(1, actual_counts.sum())

    eps = 1e-6
    psi_vals = (actual_perc - expected_perc) * np.log((actual_perc + eps) / (expected_perc + eps))
    return float(np.sum(psi_vals))


def build_quality_report(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    symbol: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    alignment_warnings: Optional[List[str]] = None,
    max_items: int = 20,
) -> Dict[str, Any]:
    """Build a data quality report for a dataset."""
    feature_cols = feature_cols or get_feature_columns(df)
    leakage = validate_feature_leakage(feature_cols)

    numeric_features = (
        df[feature_cols].select_dtypes(include=[np.number]).columns.tolist() if feature_cols else []
    )

    missing_pct = (
        df[feature_cols].isna().mean().sort_values(ascending=False)
        if feature_cols
        else pd.Series(dtype=float)
    )
    top_missing = (
        missing_pct[missing_pct > 0].head(max_items).to_dict() if not missing_pct.empty else {}
    )

    outlier_rates: Dict[str, float] = {}
    for col in numeric_features:
        outlier_rates[col] = _robust_outlier_rate(df[col])
    top_outliers = dict(sorted(outlier_rates.items(), key=lambda x: x[1], reverse=True)[:max_items])

    class_balance: Dict[str, Any] = {}
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        total = float(sum(counts.values())) or 1.0
        ratios = {str(k): float(v) / total for k, v in counts.items()}
        class_balance = {"counts": counts, "ratios": ratios}

    drift_scores: Dict[str, float] = {}
    if numeric_features and len(df) >= 50:
        split_idx = int(len(df) * 0.5)
        baseline = df.iloc[:split_idx]
        recent = df.iloc[split_idx:]
        for col in numeric_features:
            baseline_vals = baseline[col].dropna().values
            recent_vals = recent[col].dropna().values
            if baseline_vals.size < 20 or recent_vals.size < 20:
                continue
            drift_scores[col] = _psi(baseline_vals, recent_vals)

    top_drift = dict(sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)[:max_items])

    time_range: Dict[str, Any] = {}
    if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        time_range = {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
        }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": int(len(feature_cols)),
        "time_range": time_range,
        "leakage_columns": leakage,
        "missingness": {
            "total_missing_pct": float(missing_pct.mean()) if not missing_pct.empty else 0.0,
            "top_missing_pct": top_missing,
        },
        "outliers": {
            "method": "robust_zscore",
            "threshold": 4.0,
            "top_outlier_rates": top_outliers,
        },
        "class_balance": class_balance,
        "drift": {
            "method": "psi",
            "top_drift_scores": top_drift,
        },
        "alignment_warnings": alignment_warnings or [],
        "metadata": metadata or {},
    }
    return report


def save_quality_report(
    report: Dict[str, Any],
    report_dir: str = "data/reports",
    prefix: str = "dataset_quality",
) -> Path:
    """Persist a data quality report to disk."""
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    symbol = report.get("symbol") or "dataset"
    symbol_safe = str(symbol).replace("/", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{symbol_safe}_{timestamp}.json"
    full_path = report_path / filename

    with open(full_path, "w") as f:
        json.dump(report, f, indent=2)

    return full_path
