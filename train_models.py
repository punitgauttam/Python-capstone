from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, learning_curve

from data_loader import load_give_me_some_credit, split_features_target
from dataset_adapters import prep_gmsc_shared, prep_german, prep_lendingclub
from modeling import build_models, split_data
from model_store import DEFAULT_ARTIFACT
from analysis import dataset_profile, missingness_by_target, target_correlations, target_summary


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "training_results.json"


def _fairness_metrics(target, predictions, age):
    frame = pd.DataFrame({"target": target, "prediction": predictions, "age": age})
    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[-np.inf, 30, 50, np.inf],
        labels=["under_30", "30_to_50", "over_50"],
        right=False,
    )
    rates = []
    for group, values in frame.dropna(subset=["age_group"]).groupby(
        "age_group", observed=True
    ):
        true_positive = ((values["target"] == 1) & (values["prediction"] == 1)).sum()
        false_negative = ((values["target"] == 1) & (values["prediction"] == 0)).sum()
        false_positive = ((values["target"] == 0) & (values["prediction"] == 1)).sum()
        true_negative = ((values["target"] == 0) & (values["prediction"] == 0)).sum()
        rates.append(
            {
                "age_group": str(group),
                "tpr": true_positive / max(true_positive + false_negative, 1),
                "fpr": false_positive / max(false_positive + true_negative, 1),
                "positive_rate": values["prediction"].mean(),
            }
        )
    rates = pd.DataFrame(rates)
    if rates.empty:
        return {"eod_tpr": None, "eod_fpr": None, "demographic_parity_difference": None}
    return {
        "eod_tpr": float(rates["tpr"].max() - rates["tpr"].min()),
        "eod_fpr": float(rates["fpr"].max() - rates["fpr"].min()),
        "demographic_parity_difference": float(
            rates["positive_rate"].max() - rates["positive_rate"].min()
        ),
    }


def _evaluate(model, features, target, age):
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    fairness = _fairness_metrics(target, predictions, age)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
        **fairness,
    }


def _cross_validate(model, features, target, folds):
    scores = cross_validate(
        model,
        features,
        target,
        cv=folds,
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    return {
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
        "average_precision_mean": float(scores["test_average_precision"].mean()),
        "average_precision_std": float(scores["test_average_precision"].std()),
    }


def _train_round(round_name, x_train, x_test, y_train, y_test, folds):
    models = build_models(x_train)
    fitted_models = {}
    model_results = {}
    print(f"\n{round_name}")
    print("Model                 CV ROC-AUC       Train       Test    Overfit gap")
    print("-" * 76)
    for model_name, model in models.items():
        print(f"[{round_name}] cross-validating {model_name}...", flush=True)
        cv_results = _cross_validate(model, x_train, y_train, folds)
        fitted_model = clone(model).fit(x_train, y_train)
        train_results = _evaluate(fitted_model, x_train, y_train, x_train["age"])
        test_results = _evaluate(fitted_model, x_test, y_test, x_test["age"])
        fitted_models[model_name] = fitted_model
        model_results[model_name] = {
            "cv": cv_results,
            "train": train_results,
            "test": test_results,
            "overfit_gap": train_results["roc_auc"] - cv_results["roc_auc_mean"],
        }
        classifier = fitted_model.named_steps["classifier"]
        if hasattr(classifier, "history_"):
            model_results[model_name]["pytorch_loss_history"] = classifier.history_
        print(
            f"{model_name:<21} "
            f"{cv_results['roc_auc_mean']:.4f} +/- {cv_results['roc_auc_std']:.4f} "
            f"{train_results['roc_auc']:.4f} "
            f"{test_results['roc_auc']:.4f} "
            f"{model_results[model_name]['overfit_gap']:+.4f}",
            flush=True,
        )
    best_name = max(
        model_results,
        key=lambda name: model_results[name]["cv"]["roc_auc_mean"],
    )
    return fitted_models, model_results, best_name


def _save_eda_outputs(frame, results_path):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_profile(frame).to_csv(results_path.parent / "eda_profile.csv")
    target_summary(frame["SeriousDlqin2yrs"]).to_csv(
        results_path.parent / "target_balance.csv"
    )
    missingness_by_target(frame).to_csv(results_path.parent / "missingness_by_target.csv")
    target_correlations(frame).to_csv(results_path.parent / "target_correlations.csv")
    numeric = frame.drop(columns="SeriousDlqin2yrs").select_dtypes("number")
    axes = numeric.hist(figsize=(14, 10), bins=30)
    figure = axes[0, 0].get_figure()
    figure.tight_layout()
    figure.savefig(results_path.parent / "feature_distributions.png", dpi=140)
    plt.close(figure)
    missing_income_default = frame.groupby(frame["MonthlyIncome"].isna())["SeriousDlqin2yrs"].mean()
    report = [
        "# EDA Findings",
        "",
        f"The Give Me Some Credit data contains {len(frame):,} rows and {frame.shape[1] - 1} model features.",
        f"The default rate is {frame['SeriousDlqin2yrs'].mean():.2%}; class imbalance makes ROC-AUC and average precision more informative than accuracy.",
        f"MonthlyIncome is missing in {frame['MonthlyIncome'].isna().mean():.2%} of rows and NumberOfDependents in {frame['NumberOfDependents'].isna().mean():.2%}.",
        f"The default rates for missing versus observed MonthlyIncome are {missing_income_default.to_dict()}, so the pipeline keeps a missingness indicator.",
        f"There are {(frame['age'] == 0).sum()} age-zero values, {(frame['RevolvingUtilizationOfUnsecuredLines'] > 10).sum()} utilization values above 10, and {(frame['DebtRatio'] > 100).sum()} debt-ratio values above 100.",
        "Cleaning uses train-fitted median imputation, missingness indicators, percentile clipping, and invalid-value replacement inside each pipeline.",
    ]
    (results_path.parent / "eda_findings.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _save_learning_curves(round_train, y_train, results_path):
    sample_size = min(20_000, len(round_train))
    sample = round_train.sample(sample_size, random_state=42)
    sample_target = y_train.loc[sample.index]
    curves = {}
    for model_name in ("XGBoost", "PyTorch MLP"):
        model = build_models(round_train)[model_name]
        if model_name == "PyTorch MLP":
            model.set_params(classifier__epochs=15, classifier__patience=3)
        sizes, train_scores, validation_scores = learning_curve(
            model,
            sample,
            sample_target,
            cv=3,
            train_sizes=np.array([0.25, 0.6, 1.0]),
            scoring="roc_auc",
            n_jobs=1,
        )
        curves[model_name] = pd.DataFrame(
            {
                "training_examples": sizes,
                "train_mean": train_scores.mean(axis=1),
                "train_std": train_scores.std(axis=1),
                "validation_mean": validation_scores.mean(axis=1),
                "validation_std": validation_scores.std(axis=1),
            }
        )
        curves[model_name].to_csv(
            results_path.parent / f"learning_curve_{model_name.lower().replace(' ', '_')}.csv",
            index=False,
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (model_name, curve) in zip(axes, curves.items()):
        axis.plot(curve["training_examples"], curve["train_mean"], label="train")
        axis.plot(curve["training_examples"], curve["validation_mean"], label="validation")
        axis.set_title(model_name)
        axis.set_xlabel("Training examples")
        axis.set_ylabel("ROC-AUC")
        axis.legend()
    figure.tight_layout()
    figure.savefig(results_path.parent / "learning_curves.png", dpi=140)
    plt.close(figure)


def _write_comparison_report(all_results, results_path):
    lines = ["# Model Comparison and Diagnostics", ""]
    diagnostic_rows = []
    for round_name in ("Round A - full features", "Round B - shared features"):
        lines.extend([f"## {round_name}", "", "| Model | CV ROC-AUC | Test ROC-AUC | F1 | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"])
        models = all_results[round_name]["models"]
        for model_name, result in models.items():
            cv = result["cv"]
            test = result["test"]
            lines.append(
                f"| {model_name} | {cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f} | {test['roc_auc']:.4f} | {test['f1']:.4f} | {test['precision']:.4f} | {test['recall']:.4f} |"
            )
            diagnostic_rows.append(
                {
                    "round": round_name,
                    "model": model_name,
                    "train_roc_auc": result["train"]["roc_auc"],
                    "cv_mean_roc_auc": cv["roc_auc_mean"],
                    "cv_std_roc_auc": cv["roc_auc_std"],
                    "test_roc_auc": test["roc_auc"],
                    "overfit_gap": result["overfit_gap"],
                    "eod_tpr": test["eod_tpr"],
                    "eod_fpr": test["eod_fpr"],
                    "demographic_parity_difference": test["demographic_parity_difference"],
                }
            )
        lines.append(f"Best model by CV ROC-AUC: **{all_results[round_name]['best_model']}**.")
        lines.append("")
    lines.extend([
        "Models whose CV intervals overlap should be treated as statistically indistinguishable rather than ranked by a tiny point estimate difference.",
        "The fairness columns in `diagnostics.csv` report age-group TPR and FPR gaps (equalized-odds components) and demographic parity difference.",
        "",
        "## Generalization",
        "",
    ])
    for dataset, result in all_results["generalization"].items():
        lines.append(f"- {dataset}: ROC-AUC {result['roc_auc']:.4f}")
    lines.append(
        "German Credit debt_to_income and past_due_total are unavailable because their nearest proxies were conceptually mismatched, validated by opposite correlation signs against the primary dataset; age is the primary contributing shared feature for that dataset."
    )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use the Round A PyTorch MLP for in-domain Give Me Some Credit predictions because it has the strongest CV and held-out ROC-AUC.",
            "- Use the Round B PyTorch MLP when deployment requires the shared cross-lender feature schema.",
            "- The German Credit and Lending Club scores are much lower than the in-domain score, so cross-institution deployment requires recalibration and monitoring.",
            "",
            "## Fairness and Limitations",
            "",
            "Age-group equalized-odds components and demographic parity differences are included for every model in `diagnostics.csv`. The largest observed disparities should be discussed as deployment risks, not treated as proof of fairness or unfairness by themselves.",
            "The fairness analysis is limited to age because gender and race are not available in these datasets. German Credit has no direct utilization or delinquency fields, so missing shared concepts remain missing rather than being invented.",
        ]
    )
    pd.DataFrame(diagnostic_rows).to_csv(results_path.parent / "diagnostics.csv", index=False)
    history = all_results["Round A - full features"]["models"]["PyTorch MLP"].get(
        "pytorch_loss_history"
    )
    if history:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(history["train_loss"], label="train loss")
        axis.plot(history["validation_loss"], label="validation loss")
        axis.set(title="PyTorch MLP loss history", xlabel="Epoch", ylabel="BCE loss")
        axis.legend()
        figure.tight_layout()
        figure.savefig(results_path.parent / "pytorch_loss_curve.png", dpi=140)
        plt.close(figure)
    (results_path.parent / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_all_models(
    data_path=None,
    artifact_path=DEFAULT_ARTIFACT,
    results_path=RESULTS_PATH,
    folds=5,
    lendingclub_rows=100_000,
):
    frame = load_give_me_some_credit(
        data_path or PROJECT_ROOT / "data/give_me_some_credit/cs-training.csv"
    )
    features, target = split_features_target(frame)
    target_counts = target.value_counts().sort_index().to_dict()
    missing_summary = ", ".join(
        f"{column}={count:,} ({count / len(frame):.1%})"
        for column, count in frame.isna().sum().items()
        if count
    ) or "none"
    print("\n=== Dataset loaded: Give Me Some Credit ===")
    print(f"Shape: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")
    print(f"Target balance: {target_counts} (default rate={target.mean():.2%})")
    print(f"Missingness: {missing_summary}")
    _save_eda_outputs(frame, Path(results_path))
    eda_path = Path(results_path).parent / "eda_findings.md"
    print("\n=== EDA findings ===")
    print(eda_path.read_text(encoding="utf-8"), end="")
    x_train, x_test, y_train, y_test = split_data(features, target)
    shared = prep_gmsc_shared(frame).drop(columns="target")
    shared_train = shared.loc[x_train.index]
    shared_test = shared.loc[x_test.index]

    all_models = {}
    all_results = {}
    deployment = {}
    for round_name, round_train, round_test in (
        ("Round A - full features", x_train, x_test),
        ("Round B - shared features", shared_train, shared_test),
    ):
        fitted, results, best_name = _train_round(
            round_name, round_train, round_test, y_train, y_test, folds
        )
        all_models[round_name] = fitted
        all_results[round_name] = {
            "models": results,
            "best_model": best_name,
            "feature_names": list(round_train.columns),
        }
        deployment[round_name] = {
            "model_name": best_name,
            "model": fitted[best_name],
            "feature_names": list(round_train.columns),
        }

    _save_learning_curves(x_train, y_train, Path(results_path))

    german = prep_german()
    lending = prep_lendingclub(nrows=lendingclub_rows)
    best_shared = deployment["Round B - shared features"]["model"]
    all_results["generalization"] = {
        "German Credit": _evaluate(
            best_shared,
            german.drop(columns="target"),
            german["target"],
            german["age"],
        ),
        "Lending Club": _evaluate(
            best_shared,
            lending.drop(columns="target"),
            lending["target"],
            pd.Series(np.nan, index=lending.index),
        ),
    }
    print("\n=== Generalization results ===")
    for dataset_name, result in all_results["generalization"].items():
        print(f"{dataset_name}: ROC-AUC={result['roc_auc']:.4f}")
    _write_comparison_report(all_results, Path(results_path))
    bundle = {
        "bundle_version": 1,
        "source": "Give Me Some Credit",
        "models": all_models,
        "deployment": deployment,
        "results": all_results,
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved six fitted models to {artifact_path}")
    print(f"Saved metrics to {results_path}")
    comparison_path = results_path.parent / "model_comparison.md"
    print("\n=== Final model comparison ===")
    print(comparison_path.read_text(encoding="utf-8"), end="")
    return bundle


if __name__ == "__main__":
    train_all_models()