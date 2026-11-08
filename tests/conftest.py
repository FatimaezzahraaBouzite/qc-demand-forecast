import importlib

import pytest


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point every path in config at a temp dir and use fast test params."""
    from qc_demand import config

    for name in ("DATA_RAW", "DATA_PROCESSED", "MODELS_DIR", "REPORTS_DIR", "FORECASTS_DIR"):
        sub = getattr(config, name).relative_to(config.ROOT)
        monkeypatch.setattr(config, name, tmp_path / sub)
    files = {
        "DAILY_FILE": "data/processed/daily.parquet",
        "FEATURES_FILE": "data/processed/features.parquet",
        "MODEL_FILE": "models/model.joblib",
        "MODEL_META": "models/model_meta.json",
        "BACKTEST_FILE": "reports/backtest_metrics.json",
        "BACKTEST_PREDS": "reports/backtest_predictions.csv",
        "FORECAST_LOG": "forecasts/forecasts.csv",
        "MONITOR_FILE": "reports/monitoring.json",
    }
    for name, rel in files.items():
        monkeypatch.setattr(config, name, tmp_path / rel)
    config.ensure_dirs()

    params = config.load_params()
    params["backtest"] = {"n_folds": 2, "test_days": 60, "min_train_days": 365}
    params["models"]["lightgbm"]["n_estimators"] = 100
    params["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    for mod in ("qc_demand.train", "qc_demand.predict", "qc_demand.monitor", "qc_demand.ingest"):
        importlib.import_module(mod)
    return tmp_path, params
