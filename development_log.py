"""Running development and bottleneck log for the project write-up."""

DEVELOPMENT_LOG = [
    {
        "stage": "Data access",
        "what_happened": "German Credit used coded categorical values and Lending Club was too large to load in full for exploration.",
        "resolution": "Read German columns from german.doc and restricted Lending Club to needed columns and 100,000 rows.",
        "time_lost": "Moderate",
    },
    {
        "stage": "Cleaning",
        "what_happened": "Give Me Some Credit contained about 20% missing income, extreme ratios, duplicate rows, and invalid ages.",
        "resolution": "Used train-fitted median imputation, missingness indicators, percentile caps, and invalid-value flags inside pipelines.",
        "time_lost": "Moderate",
    },
    {
        "stage": "Modeling",
        "what_happened": "The neural network required scaling and early stopping; XGBoost initially hit Windows native parallelism errors during tuning.",
        "resolution": "Kept scaling inside the shared pipeline, tracked validation loss, and changed XGBoost/CV workers to single-threaded execution.",
        "time_lost": "High",
    },
    {
        "stage": "Cross-dataset",
        "what_happened": "The lenders do not expose identical fields, and German Credit has no direct utilization or delinquency count.",
        "resolution": "Defined a shared schema and kept unavailable concepts as missing rather than inventing values.",
        "time_lost": "High",
    },
    {
        "stage": "Cross-dataset",
        "what_happened": "German Credit generalization ROC-AUC was 0.4388, below random chance, before any known bug in the pipeline.",
        "resolution": "Correlation-checked debt_to_income and past_due_total proxies against their GMSC counterparts; both had opposite-sign relationships with the target (e.g. past-due proxy: -0.16 in German Credit vs +0.12 in GMSC), indicating conceptually mismatched features rather than genuine signal. Set both to missing for German Credit rather than populating them with misleading proxies. Score improved to 0.5712.",
        "time_lost": "Moderate",
    },
    {
        "stage": "Time budget",
        "what_happened": "Full CV and learning curves were computationally expensive on 120,000 training rows.",
        "resolution": "Used deterministic folds, bounded tuning grids, and a 100,000-row Lending Club evaluation sample.",
        "time_lost": "High",
    },
    {
        "stage": "Time budget",
        "what_happened": "Learning curves on the full ~96,000-row training set with a 40-epoch PyTorch model were computationally intractable (effectively appeared to hang).",
        "resolution": "Reduced the learning-curve step to a 20,000-row sample with a lighter 15-epoch/3-patience PyTorch configuration, since the diagnostic only needs to show the trend, not final accuracy.",
        "time_lost": "High",
    },
]


def development_log_frame():
    """Return the log in the table format required by the specification."""
    import pandas as pd

    return pd.DataFrame(DEVELOPMENT_LOG)
