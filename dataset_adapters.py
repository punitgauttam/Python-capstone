from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GERMAN_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "liable_maintenance",
    "telephone",
    "foreign_worker",
    "target",
]

SHARED_FEATURES = ["age", "debt_to_income", "credit_utilization", "past_due_total"]


def prep_german(path: str | Path = "data/german_credit/german.data") -> pd.DataFrame:
    """Load German Credit and create a target plus shared risk concepts."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    frame["target"] = frame["target"].map({1: 0, 2: 1}).astype("int64")
    # German Credit has no directly comparable debt-to-income, utilization, or
    # delinquency fields; keep these concepts missing rather than inventing values.
    # Three of four shared features are intentionally missing: utilization has
    # no equivalent field, while the nearest debt and past-due proxies failed
    # correlation-sign validation against the primary dataset.
    return pd.DataFrame(
        {
            "age": pd.to_numeric(frame["age"], errors="coerce"),
            "debt_to_income": np.nan,
            "credit_utilization": np.nan,
            "past_due_total": np.nan,
            "target": frame["target"],
        }
    )


def prep_lendingclub(
    path: str | Path = "data/lending_club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
    nrows: int = 100_000,
) -> pd.DataFrame:
    """Load a tractable Lending Club sample and remove unresolved outcomes."""
    columns = ["loan_status", "dti", "revol_util", "delinq_2yrs"]
    frame = pd.read_csv(path, usecols=columns, nrows=nrows, low_memory=False)
    frame = frame[frame["loan_status"].ne("Current")].copy()
    target_map = {"Charged Off": 1, "Default": 1, "Fully Paid": 0}
    frame["target"] = frame["loan_status"].map(target_map)
    frame = frame[frame["target"].notna()].copy()
    return pd.DataFrame(
        {
            "age": np.nan,
            "debt_to_income": pd.to_numeric(frame["dti"], errors="coerce"),
            "credit_utilization": pd.to_numeric(
                frame["revol_util"].astype(str).str.rstrip("%"), errors="coerce"
            ),
            "past_due_total": pd.to_numeric(frame["delinq_2yrs"], errors="coerce"),
            "target": frame["target"].astype("int64"),
        },
        index=frame.index,
    )


def prep_gmsc_shared(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate Give Me Some Credit into the same shared schema."""
    return pd.DataFrame(
        {
            "age": frame["age"],
            "debt_to_income": frame["DebtRatio"],
            "credit_utilization": frame["RevolvingUtilizationOfUnsecuredLines"],
            "past_due_total": (
                frame["NumberOfTime30-59DaysPastDueNotWorse"]
                + frame["NumberOfTime60-89DaysPastDueNotWorse"]
                + frame["NumberOfTimes90DaysLate"]
            ),
            "target": frame["SeriousDlqin2yrs"].astype("int64"),
        },
        index=frame.index,
    )


def prep_gmsc(frame: pd.DataFrame) -> pd.DataFrame:
    """Public translation-function name used by the project specification."""
    return prep_gmsc_shared(frame)
