from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocessing import CreditFeatureEngineer, build_preprocessor
from torch_model import TorchMLPClassifier


@dataclass
class ModelResult:
    name: str
    model: Pipeline
    x_test: pd.DataFrame
    y_test: pd.Series
    probabilities: object
    metrics: dict[str, float]


def split_data(features, target, test_size=0.2, random_state=42):
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_models(features):
    engineered_features = CreditFeatureEngineer().fit_transform(features)
    preprocessor = build_preprocessor(engineered_features)

    def pipeline_steps(classifier):
        return [
            ("feature_engineering", CreditFeatureEngineer()),
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ]

    return {
        "Logistic regression": Pipeline(
            steps=pipeline_steps(
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            )
        ),
        "XGBoost": Pipeline(
            steps=pipeline_steps(
                XGBClassifier(
                    eval_metric="logloss",
                    scale_pos_weight=10,
                    n_estimators=200,
                    max_depth=6,
                    n_jobs=1,
                    random_state=42,
                ),
            )
        ),
        "PyTorch MLP": Pipeline(
            steps=pipeline_steps(
                TorchMLPClassifier(
                    epochs=40,
                    batch_size=512,
                    patience=6,
                    random_state=42,
                )
            )
        ),
    }


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
    }
    return ModelResult(
        name=name,
        model=model,
        x_test=x_test,
        y_test=y_test,
        probabilities=probabilities,
        metrics=metrics,
    ), predictions


def train_and_evaluate(features, target, test_size=0.2, random_state=42):
    x_train, x_test, y_train, y_test = split_data(
        features, target, test_size=test_size, random_state=random_state
    )
    results = {}
    reports = {}
    for name, model in build_models(features).items():
        result, predictions = evaluate_model(
            name, model, x_train, x_test, y_train, y_test
        )
        results[name] = result
        reports[name] = {
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }
    return results, reports


def final_heldout_evaluation(
    x_train,
    y_train,
    x_test,
    y_test,
    selected_models,
):
    """Fit selected models on all training data and score the untouched test set."""
    available_models = build_models(x_train)
    results = {}
    reports = {}
    for name in selected_models:
        result, predictions = evaluate_model(
            name,
            available_models[name],
            x_train,
            x_test,
            y_train,
            y_test,
        )
        results[name] = result
        reports[name] = {
            "classification_report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }
    return results, reports


def curve_data(result: ModelResult):
    roc = roc_curve(result.y_test, result.probabilities)
    precision, recall, _ = precision_recall_curve(
        result.y_test, result.probabilities
    )
    return {"roc": roc, "precision": precision, "recall": recall}


def feature_importance(result: ModelResult) -> pd.Series:
    classifier = result.model.named_steps["classifier"]
    feature_names = result.model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    elif hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    else:
        first_layer = next(
            layer for layer in classifier.model_.network if hasattr(layer, "weight")
        )
        values = first_layer.weight.detach().abs().mean(dim=0).numpy()
    return pd.Series(values, index=feature_names).sort_values(ascending=False)
