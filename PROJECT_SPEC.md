# Credit Default Risk Prediction — Project Spec

## Goal
Predict the probability that a borrower will default / experience serious
delinquency, using a real-world, imperfect dataset. Build and compare
multiple models with proper cross-validation, then test how well the
best model generalizes to two other, structurally different credit
datasets (different institutions/countries/feature sets).

## Datasets

1. **Primary dataset: Give Me Some Credit** (`data/give_me_some_credit/cs-training.csv`)
   - ~150,000 rows, 10 features, binary target `SeriousDlqin2yrs`.
   - This is the dataset used for the full pipeline: EDA, cleaning,
     feature engineering, model training, cross-validation, and the
     final held-out evaluation.
   - Known issues to handle: `MonthlyIncome` ~20% missing,
     `NumberOfDependents` ~2.6% missing, extreme outliers in
     `RevolvingUtilizationOfUnsecuredLines` and `DebtRatio`, some
     `age` values of 0.
   - Target is imbalanced (~93% no-default, ~7% default) — use
     ROC-AUC / F1 / precision-recall, not raw accuracy, as the primary
     metric.

2. **Secondary dataset: German Credit** (`data/german_credit/german.data`)
   - 1000 rows, 20 features, categorical-heavy, codes explained in
     `german.doc`. No header row; columns must be assigned manually.
   - Target: recode 1→0 (good/no-default), 2→1 (bad/default).
   - Used ONLY for the generalization test at the end (not for training).

3. **Secondary dataset: Lending Club** (`data/lending_club/accepted_*.csv`)
   - Large (potentially millions of rows) — sample or restrict columns
     with `usecols`/`nrows` for tractability.
   - Target: derive from `loan_status` — map `Charged Off`/`Default` → 1,
     `Fully Paid` → 0. Drop rows with `loan_status == 'Current'`
     (outcome not yet known — including them would be leakage).
   - Used ONLY for the generalization test at the end (not for training).

## Process

### Stage 1 — EDA (on Give Me Some Credit only, ~1 hour of the 2-hour cleaning budget)
- Shape, dtypes, missingness per column (counts + %).
- Target distribution / class balance.
- Univariate distributions for all 10 features (histograms).
- Outlier detection: `.describe()`, check for impossible values
  (age = 0, utilization > 10, debt ratio > 100).
- Missingness pattern check: does `MonthlyIncome` missing correlate
  with the target or with other features?
- Correlation of features with target (informal, to build hypotheses).
- Document findings in markdown cells as you go — this is a deliverable,
  not an afterthought.

### Stage 2 — Cleaning (remainder of the 2-hour budget)
- Impute `MonthlyIncome` and `NumberOfDependents` — median imputation,
  ideally inside the pipeline (fit on train fold only) rather than on
  the whole dataset up front. Consider adding a `was_missing` flag if
  missingness looks informative from Stage 1.
- Cap/winsorize extreme outliers in `RevolvingUtilizationOfUnsecuredLines`
  and `DebtRatio` (e.g., clip at a reasonable percentile) rather than
  deleting rows outright.
- Fix or flag impossible `age` values (0 or unrealistically high).
- Document every cleaning decision with a one-line justification.
- Hard stop at 2 hours total for EDA + cleaning combined — proceed with
  an imperfect dataset if needed rather than over-cleaning.

### Stage 3 — Train/Test Split
- 80/20 split on Give Me Some Credit, stratified on the target
  (important given the class imbalance).
- Split BEFORE fitting any imputer/scaler/encoder — those must live
  inside a `Pipeline`/`ColumnTransformer` fit only on the training fold,
  to avoid leakage.

### Stage 4 — Feature Engineering
- Scale numeric features (StandardScaler or similar) — required for the
  neural network, harmless for the tree model, unnecessary but not
  harmful for the baseline.
- Optional derived features based on Stage 1 hypotheses (e.g., total
  past-due count = sum of the three "NumberOfTimeXX-YYDaysPastDue" columns).
- All transformations must be wrapped in a scikit-learn `Pipeline` so
  they apply identically to train, validation, and test folds.

### Stage 5 — Models: TWO rounds of 3 models each

**Round A — full-feature models (trained on Give Me Some Credit, all
cleaned/engineered features from Stages 1–4):**
1. **Baseline** — Logistic Regression (with `class_weight='balanced'`
   given the imbalance).
2. **Tree-based model** — Random Forest or Gradient Boosting
   (scikit-learn's `RandomForestClassifier` or `xgboost`).
3. **PyTorch neural network** — small MLP (2–3 hidden layers, 64–128
   units, dropout). Requirements:
   - Scaling MUST be applied (inside the pipeline) — unscaled inputs
     are the most common reason a NN underperforms a Random Forest.
   - Train with enough epochs — track train/val loss curves, use early
     stopping on a validation split rather than a fixed guess.
   - Use `BCEWithLogitsLoss` (binary classification) and consider
     `pos_weight` to address class imbalance.

This round answers: "how good can we do with everything we have?" It's
the primary result of the project.

**Round B — shared-feature models (trained on Give Me Some Credit,
restricted to ONLY the shared-feature subset defined in Stage 8):**
Same three model types (Logistic Regression, tree-based, PyTorch MLP),
same pipeline discipline, but using only the handful of columns that
also exist in German Credit and/or Lending Club (age, utilization,
debt-to-income, past delinquencies).

This round answers: "how much accuracy do we lose by restricting to
only the features available across lenders?" — and it's the fair,
apples-to-apples version of the model that actually gets evaluated on
German Credit and Lending Club in Stage 8 (a full-feature model can't
be evaluated on datasets missing most of its input columns).

Within each round, all three models go through the SAME preprocessing
pipeline and the SAME cross-validation folds for a fair comparison.
Report Round A and Round B results as two separate tables — do not mix
them together, since they answer different questions.

### Stage 6 — Cross-Validation
- Stratified K-Fold (5 or 10 folds) on the training split, given the
  class imbalance.
- Report MEAN and STANDARD DEVIATION of the chosen metric (ROC-AUC
  recommended) for every model — not just the mean.
- If fold-to-fold standard deviations overlap between two models,
  state explicitly that they are statistically indistinguishable
  rather than declaring a winner on a tiny gap.
- If any model's CV score looks suspiciously high, audit for leakage
  before trusting it: check that the split happened before any fitting,
  and that no feature is a disguised proxy for the target.

### Stage 7 — Final Held-Out Evaluation
- Retrain the chosen model(s) — best from Round A and best from Round B
  — on the full 80% training split.
- Evaluate once on the untouched 20% test split.
- Report ROC-AUC, F1, precision, recall, and a confusion matrix for each.
- Compare this test score to the CV mean±std — large discrepancies are
  a red flag worth discussing.

### Stage 7b — Full Diagnostic Evaluation (overfitting, underfitting, bias)

This is required for every one of the 6 models (3 in Round A, 3 in
Round B), not optional extra credit:

**Overfitting / underfitting check:**
- Compute the metric (ROC-AUC) on the TRAINING set and compare it to
  the CV mean from Stage 6.
  - Large gap (train much higher than CV) → overfitting.
  - Both low and close together → underfitting (model too simple, or
    features not informative enough).
  - Both high and close together → good fit.
- Plot a learning curve (training set size vs. train/validation score)
  for at least the tree model and the PyTorch model — this shows
  whether more data would help (still-converging curves) or whether
  the model has plateaued.
- For the PyTorch model specifically, plot the training loss vs.
  validation loss curve across epochs — a validation loss that starts
  rising while training loss keeps falling is the clearest overfitting
  signal for a neural network.

**Bias / fairness check:**
- Use `age` as the sensitive attribute (available in Give Me Some
  Credit and German Credit) — bucket into groups (e.g., under 30,
  30–50, over 50).
- Compute **Equalized Odds Difference (EOD)**: the largest gap in
  true-positive rate (and separately, false-positive rate) between age
  groups. A large EOD means the model is meaningfully more/less
  accurate for one age group than another.
- Compute **Demographic Parity Difference** alongside it: the gap in
  predicted-positive (predicted default) rate between age groups,
  regardless of actual outcome. Report both — EOD checks fairness
  conditional on the true outcome, demographic parity does not, and
  they can disagree.
- These are available via the `fairlearn` package
  (`fairlearn.metrics.equalized_odds_difference`,
  `fairlearn.metrics.demographic_parity_difference`) — add
  `fairlearn` to requirements if you use it, or compute the group-wise
  TPR/FPR/positive-rate manually with pandas if you'd rather not add
  a dependency.
- Discuss any disparity found — this belongs in the write-up's
  limitations/ethics section regardless of the result.

Summarize all of Stage 7b as one consolidated table: model | train
score | CV mean±std | test score | overfit gap | EOD | demographic
parity difference. This table is what makes the "every evaluation
possible" requirement concrete and gradable.

### Stage 8 — Cross-Dataset Generalization Test (the "three dataset" part)
- Define a small set of shared features that exist (or can be derived)
  in at least two of the three datasets. Example mapping:

  | Shared concept        | Give Me Some Credit                          | German Credit                | Lending Club   |
  |------------------------|-----------------------------------------------|-------------------------------|----------------|
  | age                    | `age`                                         | `Age` (attribute 13)          | not available  |
  | credit utilization     | `RevolvingUtilizationOfUnsecuredLines`        | not directly available        | `revol_util`   |
  | debt-to-income proxy   | `DebtRatio`                                   | not directly available        | `dti`          |
  | past delinquencies     | sum of the three "past due" columns           | not directly available        | `delinq_2yrs`  |
  | target (default=1)     | `SeriousDlqin2yrs`                            | recoded `target` (1=bad)      | `loan_status` mapped |

- Write one small "translation" function per dataset (`prep_gmsc`,
  `prep_german`, `prep_lendingclub`) that outputs only these shared
  columns — do NOT merge the three datasets into one table.
- Take the single best model's ALREADY-FITTED pipeline from Stage 7
  (trained only on Give Me Some Credit) and use `.transform()` +
  `.predict()`/`.predict_proba()` on the shared-feature versions of
  German Credit and Lending Club — do not refit anything on them.
- Report ROC-AUC on each of the two secondary datasets alongside the
  in-domain test score from Stage 7. A performance drop is an expected,
  legitimate finding — report it honestly rather than treating it as
  a failure.
- Optional (if time allows): also train a second, lightweight version
  of the best model using ONLY the shared feature subset on Give Me
  Some Credit, so the "apples to apples" comparison across all three
  datasets is on equal footing (in addition to, not instead of, the
  full-feature model from Stage 5–7).

## Deliverables / Write-up structure
1. Introduction — problem statement, why three datasets, why this matters.
2. EDA findings (Give Me Some Credit) — key plots + written observations.
3. Cleaning decisions — table of what was done and why.
4. Methodology — pipeline structure, Round A vs Round B models, CV
   strategy, metrics chosen and why (imbalance justification, why EOD
   and demographic parity were added).
5. Results — Round A CV table (mean ± std per model per metric), Round B
   CV table, held-out test results, confusion matrices.
6. Full diagnostic table (Stage 7b) — train/CV/test scores, overfit gap,
   EOD, demographic parity difference, for all 6 models. Discuss which
   models overfit, which underfit, and whether any showed meaningful
   age-group disparity.
7. Generalization results — shared-feature table, Round B model scores
   on German Credit and Lending Club, honest discussion of the
   performance drop and likely causes (population differences, feature
   availability differences, era/currency differences for German
   Credit, etc.).
8. Recommendation — which model to use in-domain (from Round A), which
   to use if cross-lender deployment is required (best Round B model),
   and an honest caveat about deploying to a different institution's data.
9. Limitations — cleaning shortcuts taken under the time budget, features
   that couldn't be matched across datasets, sample size differences,
   fairness caveats (age was the only sensitive attribute available;
   other protected attributes like gender/race were not present in
   these datasets and so could not be checked).
10. Bottlenecks & Development Log — see dedicated section below.

## Bottlenecks & Development Log

Keep this as a running log throughout the project, not something
written from memory at the end — add an entry whenever something
takes longer than expected, breaks, or forces a decision. This becomes
part of the write-up and shows the professor the real process, not
just the polished result.

**Format per entry:**
```
[Stage] — [What happened] — [How it was resolved] — [Time lost, if notable]
```

**Categories to cover (fill in with your own real entries as you go):**

- **Data access bottlenecks** — e.g., Kaggle API requiring account
  setup + `kaggle.json` credentials + separately accepting competition
  rules before download would work; German Credit's undocumented
  code-based categorical format requiring the separate `.doc` file to
  decode.
- **Cleaning bottlenecks** — e.g., deciding how to handle the ~20%
  missing `MonthlyIncome` in Give Me Some Credit, or where to cap the
  extreme `RevolvingUtilizationOfUnsecuredLines` outliers, and why that
  threshold was chosen over another.
- **Modeling bottlenecks** — e.g., PyTorch model underperforming before
  scaling was correctly placed inside the pipeline; number of epochs
  tuned after inspecting the loss curve; class imbalance requiring
  `class_weight`/`pos_weight` adjustments rather than default settings.
- **Cross-dataset bottlenecks** — e.g., features available in one
  dataset but not another, forcing the shared-feature set to shrink;
  Lending Club's large file size requiring column/row subsetting;
  German Credit having no direct debt-to-income equivalent.
- **Time budget notes** — record actual time spent on EDA+cleaning vs.
  the 2-hour target, and what was deliberately left unfixed as a result.

**Why this section matters for the write-up:** it directly documents
that fold-variance, leakage, and scaling checks (the pitfalls called
out in the assignment) were actively watched for during development,
not just mentioned as a formality in the methodology section.

## Key pitfalls to avoid (do not skip these checks)
- Do not fit imputers/scalers/encoders on the full dataset before
  splitting — always fit on the training fold only.
- Do not judge models on tiny CV score differences without checking
  fold standard deviation.
- Do not merge all three datasets into a single training table.
- Do not include `loan_status == 'Current'` rows in Lending Club (no
  known outcome yet).
- Do not skip scaling before training the PyTorch model.
- Do not train separate models per dataset — models are trained only
  on Give Me Some Credit (Round A: full features, Round B: shared
  features); German Credit and Lending Club are evaluation-only.
- Do not report only a single accuracy/AUC number per model — the
  Stage 7b diagnostic table (train/CV/test/overfit gap/fairness) is
  required for every model, not just the winner.
- Do not skip the fairness check just because the dataset "looks"
  fine — compute EOD and demographic parity difference regardless of
  expectation, and report the actual numbers.
