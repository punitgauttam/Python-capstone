from __future__ import annotations

import pandas as pd

from data_loader import TARGET_COLUMN


def dataset_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize type, missingness, cardinality, and distribution for every column."""
    summary = pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing_count": frame.isna().sum(),
            "missing_rate": frame.isna().mean(),
            "unique_count": frame.nunique(dropna=True),
        }
    )
    summary["mean"] = frame.mean(numeric_only=True)
    summary["median"] = frame.median(numeric_only=True)
    summary["minimum"] = frame.min(numeric_only=True)
    summary["maximum"] = frame.max(numeric_only=True)
    return summary


def target_summary(target: pd.Series) -> pd.DataFrame:
    """Return counts and rates for the binary target."""
    counts = target.value_counts().sort_index().rename("count")
    summary = counts.to_frame()
    summary["rate"] = summary["count"] / len(target)
    return summary


def outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Find numeric columns with observations outside the 1.5-IQR fences."""
    numeric = frame.select_dtypes(include="number")
    first_quartile = numeric.quantile(0.25)
    third_quartile = numeric.quantile(0.75)
    spread = third_quartile - first_quartile
    lower = first_quartile - 1.5 * spread
    upper = third_quartile + 1.5 * spread
    counts = ((numeric.lt(lower)) | (numeric.gt(upper))).sum()
    return pd.DataFrame(
        {
            "outlier_count": counts,
            "outlier_rate": counts / len(frame),
            "lower_fence": lower,
            "upper_fence": upper,
        }
    ).sort_values("outlier_count", ascending=False)


def target_correlations(frame: pd.DataFrame) -> pd.Series:
    """Rank numeric features by absolute Pearson correlation with the target."""
    correlations = frame.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def duplicate_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Count exact duplicates and duplicated index values."""
    return {
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_indices": int(frame.index.duplicated().sum()),
    }


def missingness_by_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare missingness rates for each feature across target classes."""
    features = frame.drop(columns=TARGET_COLUMN)
    result = features.isna().groupby(frame[TARGET_COLUMN]).mean().T
    result.columns = [f"target_{column}_missing_rate" for column in result.columns]
    return result.sort_values(result.columns[-1], ascending=False)


def feature_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag constant, near-constant, and high-cardinality columns."""
    rows = []
    for column in frame.columns:
        counts = frame[column].value_counts(dropna=False, normalize=True)
        rows.append(
            {
                "feature": column,
                "constant": frame[column].nunique(dropna=False) <= 1,
                "dominant_value_rate": counts.iloc[0],
                "near_constant": counts.iloc[0] >= 0.99,
                "high_cardinality": frame[column].nunique(dropna=True) > 0.5 * len(frame),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def timing_leakage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Record the manual prediction-time availability audit for every feature."""
    return pd.DataFrame(
        {
            "feature": frame.drop(columns=TARGET_COLUMN).columns,
            "available_at_application": True,
            "reason": "Borrower application and credit-history field; no post-outcome field is present.",
        }
    ).set_index("feature")


def binned_default_rates(
    frame: pd.DataFrame, feature: str, bins: int = 10
) -> pd.DataFrame:
    """Calculate default rate and sample count across quantile bins."""
    if feature not in frame.columns:
        raise KeyError(f"Unknown feature: {feature}")
    groups = pd.qcut(frame[feature], q=bins, duplicates="drop")
    return (
        frame.groupby(groups, observed=False)[TARGET_COLUMN]
        .agg(default_rate="mean", sample_count="size")
        .reset_index()
    )
