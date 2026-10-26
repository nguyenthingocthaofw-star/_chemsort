"""Filesystem layout of the ML project, resolved from this file's location.

Every new pipeline uses these absolute paths, so commands work from any
working directory. The legacy scripts in ``pipelines/legacy`` still expect to
be run from the ``ml/`` root.
"""

from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
BUILD_DIR = DATA_DIR / "build"

MODELS_DIR = ML_ROOT / "models"
REPORTS_DIR = ML_ROOT / "reports"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
COMPARISON_DIR = REPORTS_DIR / "comparison"

MANIFESTS = {
    "v1": DATA_DIR / "manifest.csv",
    "v2": DATA_DIR / "manifest_v2.csv",
}

FLUTTER_ROOT = ML_ROOT.parent
FLUTTER_ASSETS_DIR = FLUTTER_ROOT / "assets" / "chemsort"
DATA_BUNDLE_ZIP = FLUTTER_ROOT / "ChemSort_Data.zip"


def run_model_dir(run_name: str) -> Path:
    return MODELS_DIR / run_name


def run_report_dir(run_name: str) -> Path:
    return EXPERIMENTS_DIR / run_name


def evaluation_dir(run_name: str, artifact: str, split: str) -> Path:
    return EXPERIMENTS_DIR / run_name / artifact / split
