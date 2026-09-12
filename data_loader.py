from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/give_me_some_credit/cs-training.csv")
TARGET_COLUMN = "SeriousDlqin2yrs"


def load_give_me_some_credit(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the labeled Give Me Some Credit training data."""
    frame = pd.read_csv(path, index_col=0)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} in {path}")
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and the binary default target."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}")
    features = frame.drop(columns=TARGET_COLUMN).copy()
    target = frame[TARGET_COLUMN].astype("int64")
    return features, target
