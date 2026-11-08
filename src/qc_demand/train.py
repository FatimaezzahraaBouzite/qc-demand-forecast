"""Backtest all models, log everything to MLflow, refit the winner on all data."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import mlflow
import pandas as pd

from . import config
from .backtest import run_backtest, summarize
from .features import build_features, feature_columns, model_ready
from .models import make_model

log = logging.getLogger(__name__)


def load_frame(params: dict, until: str | None = None) -> pd.DataFrame:
    daily = pd.read_parquet(config.DAILY_FILE)
    feats = build_features(daily, params)
    feats.to_parquet(config.FEATURES_FILE)
    frame = model_ready(feats, params)
    return frame.loc[:until] if until else frame


def run(params: dict, until: str | None = None) -> dict:
    """`until` holds data back (e.g. train through June, then replay H2 as 'live')."""
    config.ensure_dirs()
    frame = load_frame(params, until)
    mp = params["mlflow"]
    mlflow.set_tracking_uri(mp["tracking_uri"])
    mlflow.set_experiment(mp["experiment"])

    with mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M}") as parent:
        mlflow.log_params({"n_days": len(frame), "data_start": str(frame.index.min().date()),
                           "data_end": str(frame.index.max().date()),
                           **{f"bt_{k}": v for k, v in params["backtest"].items()}})
        fold_metrics, preds = run_backtest(frame, params)
        summary = summarize(fold_metrics)
        log.info("Backtest summary:\n%s", summary)

        # one nested run per candidate, so the MLflow UI shows a clean comparison
        for name, g in fold_metrics.groupby("model"):
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"model": name, **(params["models"].get(name) or {})})
                for metric in ("mape", "mae", "rmse", "bias"):
                    mlflow.log_metric(f"bt_{metric}_mean", g[metric].mean())
                    for _, r in g.iterrows():
                        mlflow.log_metric(f"bt_{metric}", r[metric], step=int(r["fold"]))

        sel = params["selection_metric"]
        means = fold_metrics.groupby("model")[sel].mean()
        best = means.idxmin()
        baseline = means.get("seasonal_naive")
        log.info("Selected %s (%s=%.3f, baseline=%.3f)", best, sel, means[best], baseline)

        model = make_model(best, params).fit(frame)
        joblib.dump(model, config.MODEL_FILE)
        meta = {
            "model": best,
            "selection_metric": sel,
            "backtest_mean": {m: round(float(v), 4) for m, v in means.items()},
            "skill_vs_naive_pct": round(float((1 - means[best] / baseline) * 100), 2)
            if baseline else None,
            "features": feature_columns(params),
            "trained_until": str(frame.index.max().date()),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": parent.info.run_id,
        }
        if best == "lightgbm":
            imp = model.feature_importance()
            meta["top_features"] = imp.head(10).round(1).to_dict()
            imp.to_csv(config.REPORTS_DIR / "feature_importance.csv", header=["gain"])
            mlflow.log_artifact(str(config.REPORTS_DIR / "feature_importance.csv"))

        config.MODEL_META.write_text(json.dumps(meta, indent=2))
        report = {"summary": {m: {k: round(float(v), 4) for k, v in g[["mape", "mae", "rmse",
                                                                       "bias"]].mean().items()}
                              for m, g in fold_metrics.groupby("model")},
                  "folds": fold_metrics.to_dict(orient="records"),
                  "selected": best}
        config.BACKTEST_FILE.write_text(json.dumps(report, indent=2))
        preds.to_csv(config.BACKTEST_PREDS, index=False)

        mlflow.log_param("selected_model", best)
        mlflow.log_metric(f"selected_{sel}", float(means[best]))
        for f in (config.BACKTEST_FILE, config.BACKTEST_PREDS, config.MODEL_META,
                  config.MODEL_FILE):
            mlflow.log_artifact(str(f))
    return meta
