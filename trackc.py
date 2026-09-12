from __future__ import annotations

import argparse
import json

from model_store import DEFAULT_ARTIFACT
from predictor import load_or_train_bundle, predict_default_probability


INPUT_FIELDS = [
	("RevolvingUtilizationOfUnsecuredLines", float),
	("age", float),
	("NumberOfTime30-59DaysPastDueNotWorse", float),
	("DebtRatio", float),
	("MonthlyIncome", float),
	("NumberOfOpenCreditLinesAndLoans", float),
	("NumberOfTimes90DaysLate", float),
	("NumberRealEstateLoansOrLines", float),
	("NumberOfTime60-89DaysPastDueNotWorse", float),
	("NumberOfDependents", float),
]


def read_interactive_borrower():
	borrower = {}
	print("Enter borrower information. Press Enter for an unknown numeric value.")
	for field, converter in INPUT_FIELDS:
		value = input(f"{field}: ").strip()
		borrower[field] = None if value == "" else converter(value)
	return borrower


def main(argv=None):
	parser = argparse.ArgumentParser(description="Credit default risk predictor")
	parser.add_argument("--input", help="JSON file containing one borrower record.")
	parser.add_argument(
		"--artifact",
		default=str(DEFAULT_ARTIFACT),
		help="Cached model bundle path.",
	)
	parser.add_argument(
		"--train",
		action="store_true",
		help="Force training and replace the cached artifact.",
	)
	args = parser.parse_args(argv)

	bundle, trained_now = load_or_train_bundle(
		artifact_path=args.artifact,
		force_retrain=args.train,
	)
	print("Model trained and cached." if trained_now else "Cached model loaded; no training needed.")

	if args.input:
		with open(args.input, "r", encoding="utf-8") as file:
			borrower = json.load(file)
	else:
		borrower = read_interactive_borrower()

	prediction = predict_default_probability(borrower, bundle=bundle)
	print(json.dumps(prediction, indent=2))
	return prediction


if __name__ == "__main__":
	main()
