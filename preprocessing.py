import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain features without using the target or future information."""

    def fit(self, features, target=None):
        return self

    def transform(self, features):
        transformed = features.copy()
        past_due_columns = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
        ]
        if all(column in transformed for column in past_due_columns):
            transformed["PastDueTotal"] = transformed[past_due_columns].sum(axis=1)
            transformed["SeriousLateIndicator"] = (
                transformed["NumberOfTimes90DaysLate"] > 0
            ).astype(int)
        if "MonthlyIncome" in transformed:
            transformed["IncomeMissing"] = transformed["MonthlyIncome"].isna().astype(int)
            if "NumberOfDependents" in transformed:
                transformed["DependentsMissing"] = transformed["NumberOfDependents"].isna().astype(int)
                transformed["IncomePerDependent"] = transformed["MonthlyIncome"] / (
                    transformed["NumberOfDependents"].fillna(0) + 1
                )
            transformed["LogMonthlyIncome"] = np.log1p(transformed["MonthlyIncome"].clip(lower=0))
        return transformed


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Fit percentile caps on training data and apply them to later partitions."""

    def __init__(self, columns=None, lower=0.01, upper=0.99):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def fit(self, features, target=None):
        columns = (
            features.select_dtypes(include=np.number).columns
            if self.columns is None
            else self.columns
        )
        self.bounds_ = {
            column: (
                features[column].quantile(self.lower),
                features[column].quantile(self.upper),
            )
            for column in columns
            if column in features
        }
        return self

    def transform(self, features):
        transformed = features.copy()
        for column, (lower, upper) in self.bounds_.items():
            transformed[column] = transformed[column].clip(lower=lower, upper=upper)
        return transformed


def replace_invalid_values(features):
    """Mark deterministic domain violations as missing for pipeline imputation."""
    cleaned = features.copy()
    if "age" in cleaned:
        cleaned.loc[(cleaned["age"] < 18) | (cleaned["age"] > 100), "age"] = np.nan
    nonnegative_columns = cleaned.select_dtypes(include=np.number).columns
    for column in nonnegative_columns:
        if column in cleaned:
            cleaned.loc[cleaned[column] < 0, column] = np.nan
    return cleaned


def build_preprocessor(features):
    """Create a train-only cleaning, imputation, and scaling pipeline."""
    numeric_features = list(features.columns)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "validity",
                            FunctionTransformer(
                                replace_invalid_values,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("outlier_caps", QuantileClipper()),
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )
