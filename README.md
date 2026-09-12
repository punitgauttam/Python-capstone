# Credit Default Risk Prediction

This project predicts the probability that a borrower will experience serious delinquency. It uses the Give Me Some Credit dataset for training and evaluates cross-dataset generalization on German Credit and Lending Club data.

The project trains and compares six models:

- Round A: Logistic Regression, XGBoost, and a PyTorch MLP using the full Give Me Some Credit feature set.
- Round B: Logistic Regression, XGBoost, and a PyTorch MLP using shared features for cross-lender evaluation.

The fitted pipelines, evaluation metrics, fairness diagnostics, learning curves, and EDA outputs are stored in `artifacts/`. The tracked `artifacts/credit_models.joblib` file means deployment can load the fitted models without retraining.

## Setup

Clone the repository and enter the project directory:

```powershell
git clone <your-repository-url>
cd restored_project
```

Create and activate a virtual environment if desired, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
```

## Data Download

The datasets are intentionally excluded from Git because of their size and distribution terms. Configure Kaggle credentials before downloading the Kaggle datasets:

1. Install or configure a Kaggle API token for your user account.
2. Accept the Give Me Some Credit competition rules on Kaggle.
3. Run:

```powershell
python data_download.py
```

The downloader saves the primary dataset under `data/give_me_some_credit/`, German Credit under `data/german_credit/`, and Lending Club under `data/lending_club/`.

## Train And Evaluate

Run the full workflow:

```powershell
python train_models.py
```

This performs EDA, leakage-safe preprocessing, five-fold stratified cross-validation, held-out evaluation, six-model diagnostics, age fairness checks, learning-curve generation, and cross-dataset evaluation. It writes the fitted artifact to `artifacts/credit_models.joblib` and metrics/report files to `artifacts/`.

## Predict

Use a JSON borrower record:

```powershell
python trackc.py --input example_borrower.json
```

Or enter borrower values interactively:

```powershell
python trackc.py
```

Prediction loads `artifacts/credit_models.joblib` directly. It does not retrain unless `--train` is explicitly supplied.

## Important Files

- `PROJECT_SPEC.md`: complete project requirements and write-up structure.
- `train_models.py`: training, evaluation, diagnostics, and artifact generation entry point.
- `trackc.py`: command-line and interactive prediction entry point.
- `predictor.py`: artifact loading and prediction API.
- `preprocessing.py`: schema-independent validity checks, clipping, imputation, and scaling.
- `artifacts/model_comparison.md`: model comparison, recommendations, generalization results, and limitations.
- `artifacts/diagnostics.csv`: consolidated train/CV/test, overfitting, and age fairness metrics.

Do not commit `data/`, Kaggle credentials, or virtual-environment folders. The trained model artifact is intentionally tracked so a cloned repository can serve predictions without retraining.
