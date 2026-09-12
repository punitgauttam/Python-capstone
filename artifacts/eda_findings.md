# EDA Findings

The Give Me Some Credit data contains 150,000 rows and 10 model features.
The default rate is 6.68%; class imbalance makes ROC-AUC and average precision more informative than accuracy.
MonthlyIncome is missing in 19.82% of rows and NumberOfDependents in 2.62%.
The default rates for missing versus observed MonthlyIncome are {False: 0.06948590243537403, True: 0.05613669234132723}, so the pipeline keeps a missingness indicator.
There are 1 age-zero values, 241 utilization values above 10, and 24380 debt-ratio values above 100.
Cleaning uses train-fitted median imputation, missingness indicators, percentile clipping, and invalid-value replacement inside each pipeline.
