from __future__ import annotations

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = PROJECT_ROOT / "artifacts" / "credit_models.joblib"


def save_bundle(bundle, path: str | Path = DEFAULT_ARTIFACT):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def load_bundle(path: str | Path = DEFAULT_ARTIFACT):
    return joblib.load(path)


def bundle_exists(path: str | Path = DEFAULT_ARTIFACT) -> bool:
    return Path(path).is_file()
