from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import learning_curve


def age_groups(age):
    return pd.cut(
        age,
        bins=[-np.inf, 30, 50, np.inf],
        labels=["under_30", "30_to_50", "over_50"],
        right=False,
    )


def fairness_metrics(target, predictions, age):
    """Compute demographic parity and equalized-odds gaps by age group."""
    groups = age_groups(pd.Series(age, index=target.index))
    rows = []
    for group in groups.dropna().unique():
        mask = groups == group
        actual = target[mask]
        predicted = pd.Series(predictions, index=target.index)[mask]
        matrix = confusion_matrix(actual, predicted, labels=[0, 1])
        true_negative, false_positive, false_negative, true_positive = matrix.ravel()
        rows.append(
            {
                "age_group": group,
                "true_positive_rate": true_positive / max(true_positive + false_negative, 1),
                "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
                "predicted_positive_rate": predicted.mean(),
            }
        )
    group_metrics = pd.DataFrame(rows).set_index("age_group")
    return {
        "group_metrics": group_metrics,
        "eod_tpr": group_metrics["true_positive_rate"].max()
        - group_metrics["true_positive_rate"].min(),
        "eod_fpr": group_metrics["false_positive_rate"].max()
        - group_metrics["false_positive_rate"].min(),
        "demographic_parity_difference": group_metrics["predicted_positive_rate"].max()
        - group_metrics["predicted_positive_rate"].min(),
    }


def diagnostic_row(name, model, x_train, y_train, x_test, y_test, cv_mean, cv_std):
    """Fit one model and return train/test, overfit, and fairness diagnostics."""
    model.fit(x_train, y_train)
    train_probabilities = model.predict_proba(x_train)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    fairness = fairness_metrics(y_test, test_predictions, x_test["age"])
    train_score = roc_auc_score(y_train, train_probabilities)
    test_score = roc_auc_score(y_test, test_probabilities)
    return {
        "model": name,
        "train_roc_auc": train_score,
        "cv_mean_roc_auc": cv_mean,
        "cv_std_roc_auc": cv_std,
        "test_roc_auc": test_score,
        "overfit_gap": train_score - cv_mean,
        "eod_tpr": fairness["eod_tpr"],
        "eod_fpr": fairness["eod_fpr"],
        "demographic_parity_difference": fairness["demographic_parity_difference"],
        "fairness_groups": fairness["group_metrics"],
        "fitted_model": model,
    }


def learning_curve_data(model, features, target, folds=3, random_state=42):
    """Return learning-curve arrays for train and validation ROC-AUC."""
    sizes, train_scores, validation_scores = learning_curve(
        model,
        features,
        target,
        cv=folds,
        scoring="roc_auc",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "training_examples": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "validation_mean": validation_scores.mean(axis=1),
            "validation_std": validation_scores.std(axis=1),
        }
    )
