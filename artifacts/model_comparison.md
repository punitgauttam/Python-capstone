# Model Comparison and Diagnostics

## Round A - full features

| Model | CV ROC-AUC | Test ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.8560 +/- 0.0048 | 0.8621 | 0.3430 | 0.2224 | 0.7491 |
| XGBoost | 0.8323 +/- 0.0044 | 0.8391 | 0.3819 | 0.2832 | 0.5865 |
| PyTorch MLP | 0.8620 +/- 0.0058 | 0.8657 | 0.3358 | 0.2138 | 0.7820 |
Best model by CV ROC-AUC: **PyTorch MLP**.

## Round B - shared features

| Model | CV ROC-AUC | Test ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.8493 +/- 0.0040 | 0.8545 | 0.3331 | 0.2149 | 0.7401 |
| XGBoost | 0.8282 +/- 0.0037 | 0.8375 | 0.3652 | 0.2584 | 0.6224 |
| PyTorch MLP | 0.8523 +/- 0.0048 | 0.8591 | 0.3218 | 0.2023 | 0.7855 |
Best model by CV ROC-AUC: **PyTorch MLP**.

Models whose CV intervals overlap should be treated as statistically indistinguishable rather than ranked by a tiny point estimate difference.
The fairness columns in `diagnostics.csv` report age-group TPR and FPR gaps (equalized-odds components) and demographic parity difference.

## Generalization

- German Credit: ROC-AUC 0.5712
- Lending Club: ROC-AUC 0.5119
German Credit debt_to_income and past_due_total are unavailable because their nearest proxies were conceptually mismatched, validated by opposite correlation signs against the primary dataset; age is the primary contributing shared feature for that dataset.

## Recommendation

- Use the Round A PyTorch MLP for in-domain Give Me Some Credit predictions because it has the strongest CV and held-out ROC-AUC.
- Use the Round B PyTorch MLP when deployment requires the shared cross-lender feature schema.
- The German Credit and Lending Club scores are much lower than the in-domain score, so cross-institution deployment requires recalibration and monitoring.

## Fairness and Limitations

Age-group equalized-odds components and demographic parity differences are included for every model in `diagnostics.csv`. The largest observed disparities should be discussed as deployment risks, not treated as proof of fairness or unfairness by themselves.
The fairness analysis is limited to age because gender and race are not available in these datasets. German Credit has no direct utilization or delinquency fields, so missing shared concepts remain missing rather than being invented.
