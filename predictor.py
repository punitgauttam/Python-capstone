from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_store import DEFAULT_ARTIFACT, bundle_exists, load_bundle


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "PyTorch MLP"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "give_me_some_credit" / "cs-training.csv"


def train_deployment_bundle(
    data_path=DEFAULT_DATA_PATH,
    artifact_path=DEFAULT_ARTIFACT,
    model_name=DEFAULT_MODEL_NAME,
):
    """Explicitly retrain and persist all six project models."""
    from train_models import train_all_models

    return train_all_models(data_path=data_path, artifact_path=artifact_path)


def load_or_train_bundle(
    artifact_path=DEFAULT_ARTIFACT,
    data_path=DEFAULT_DATA_PATH,
    model_name=DEFAULT_MODEL_NAME,
    force_retrain=False,
):
    """Load fitted models; retraining is only allowed when explicitly requested."""
    if force_retrain:
        return train_deployment_bundle(data_path, artifact_path, model_name), True
    if bundle_exists(artifact_path):
        return load_bundle(artifact_path), False
    raise FileNotFoundError(
        f"No fitted model bundle found at {artifact_path}. "
        "Run train_models.py once before deploying."
    )


def predict_default_probability(borrower: dict, bundle=None, artifact_path=DEFAULT_ARTIFACT):
    """Return a default probability and a simple risk decision."""
    if bundle is None:
        bundle = load_bundle(artifact_path)
    deployment = bundle.get("deployment", {}).get(
        "Round A - full features", bundle
    )
    features = pd.DataFrame([borrower]).reindex(columns=deployment["feature_names"])
    probability = float(deployment["model"].predict_proba(features)[0, 1])
    threshold = float(deployment.get("threshold", 0.5))
    return {
        "model": deployment["model_name"],
        "default_probability": probability,
        "threshold": threshold,
        "decision": "higher-risk" if probability >= threshold else "lower-risk",
    }
