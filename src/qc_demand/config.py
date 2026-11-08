"""Paths and parameter loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("QC_DEMAND_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FORECASTS_DIR = ROOT / "forecasts"

DAILY_FILE = DATA_PROCESSED / "daily.parquet"
FEATURES_FILE = DATA_PROCESSED / "features.parquet"
MODEL_FILE = MODELS_DIR / "model.joblib"
MODEL_META = MODELS_DIR / "model_meta.json"
BACKTEST_FILE = REPORTS_DIR / "backtest_metrics.json"
BACKTEST_PREDS = REPORTS_DIR / "backtest_predictions.csv"
FORECAST_LOG = FORECASTS_DIR / "forecasts.csv"
MONITOR_FILE = REPORTS_DIR / "monitoring.json"


def load_params(path: Path | None = None) -> dict:
    path = path or ROOT / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FORECASTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
