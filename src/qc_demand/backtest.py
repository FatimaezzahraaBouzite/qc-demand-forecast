"""Rolling-origin backtest with an expanding training window.

Fold k trains on everything before its test block, then forecasts the block one day
ahead at a time. Test blocks are the last n_folds * test_days days, so each model is
judged on every season of the final year.
"""
from __future__ import annotations

import pandas as pd

from .features import TARGET
from .models import make_model, metrics


def make_folds(index: pd.DatetimeIndex, n_folds: int, test_days: int, min_train_days: int):
    folds = []
    end = len(index)
    for k in range(n_folds, 0, -1):
        test_start = end - k * test_days
        test_end = test_start + test_days
        if test_start < min_train_days:
            raise ValueError("Not enough history for the requested folds")
        folds.append((index[:test_start], index[test_start:test_end]))
    return folds


def run_backtest(frame: pd.DataFrame, params: dict, model_names=None):
    """Returns (per-fold metrics DataFrame, predictions DataFrame)."""
    bp = params["backtest"]
    model_names = model_names or list(params["models"])
    folds = make_folds(frame.index, bp["n_folds"], bp["test_days"], bp["min_train_days"])
    rows, preds = [], []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train = frame.loc[train_idx]
        for name in model_names:
            model = make_model(name, params).fit(train)
            p = model.predict(frame.loc[: test_idx.max()], test_idx)
            m = metrics(frame.loc[test_idx, TARGET], p)
            rows.append({"model": name, "fold": fold,
                         "test_start": str(test_idx.min().date()),
                         "test_end": str(test_idx.max().date()), **m})
            preds.append(pd.DataFrame({"date": test_idx, "model": name, "fold": fold,
                                       "actual": frame.loc[test_idx, TARGET].to_numpy(),
                                       "pred": p.to_numpy()}))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (fold_metrics.groupby("model")[["mape", "mae", "rmse", "bias"]]
            .agg(["mean", "std"]).round(3))
