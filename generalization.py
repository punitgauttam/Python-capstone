from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from dataset_adapters import prep_german, prep_lendingclub


def evaluate_cross_dataset(model, german=None, lendingclub=None):
    """Score an already-fitted shared-feature model without refitting it."""
    evaluations = {}
    for name, dataset in (("German Credit", german), ("Lending Club", lendingclub)):
        if dataset is None or dataset.empty:
            continue
        features = dataset.drop(columns="target")
        probabilities = model.predict_proba(features)[:, 1]
        evaluations[name] = {
            "rows": len(dataset),
            "default_rate": dataset["target"].mean(),
            "roc_auc": roc_auc_score(dataset["target"], probabilities),
        }
    return pd.DataFrame(evaluations).T


def load_generalization_datasets(german_path=None, lendingclub_path=None, nrows=100_000):
    """Load both evaluation-only datasets through their translation functions."""
    german = prep_german(german_path) if german_path else prep_german()
    lendingclub = (
        prep_lendingclub(lendingclub_path, nrows=nrows)
        if lendingclub_path
        else prep_lendingclub(nrows=nrows)
    )
    return german, lendingclub
