"""Post-production monitoring: forecast accuracy against actuals, plus input drift.

Two signals, because they fail differently:
- performance: rolling MAPE of logged forecasts vs actuals, compared with the backtest
  MAPE the model was selected on (alert when it degrades by mape_alert_ratio);
- drift: PSI of the temperature seen in production vs training. Drift can warn before
  accuracy drops (e.g. a record cold snap outside the training range).
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config
from .features import TARGET

log = logging.getLogger(__name__)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the reference sample."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def seasonal_reference(train: pd.Series, window: pd.DatetimeIndex, days: int = 15) -> pd.Series:
    """Training values from the same time of year as `window` (+/- `days`).

    Comparing December temperatures with a whole year of training data would always
    'drift'; the question is whether this December is unusual for a December.
    """
    tr_doy = train.index.day_of_year.to_numpy()
    keep = np.zeros(len(train), dtype=bool)
    for d in np.unique(window.day_of_year):
        dist = np.abs(tr_doy - d)
        keep |= np.minimum(dist, 365 - dist) <= days
    return train[keep]


def run(params: dict) -> dict:
    mp = params["monitoring"]
    meta = json.loads(config.MODEL_META.read_text())
    daily = pd.read_parquet(config.DAILY_FILE)
    fc = pd.read_csv(config.FORECAST_LOG, parse_dates=["target_date"])
    fc = fc.sort_values("issued_at").drop_duplicates("target_date", keep="last")
    df = fc.set_index("target_date").join(daily[[TARGET]], how="left").dropna(subset=[TARGET])
    if df.empty:
        report = {"status": "no_actuals_yet", "n_forecasts": int(len(fc))}
        config.MONITOR_FILE.write_text(json.dumps(report, indent=2))
        return report

    df["ape"] = (df["pred_mw"] - df[TARGET]).abs() / df[TARGET] * 100
    w = mp["window_days"]
    recent = df.tail(w)
    bt_mape = meta["backtest_mean"][meta["model"]]
    recent_mape = float(recent["ape"].mean())

    train_temps = daily.loc[: meta["trained_until"], "temp_mean"].dropna()
    ref = seasonal_reference(train_temps, recent.index)
    cur = recent["temp_mean"].dropna().to_numpy()
    # few points in a 14-day window -> few bins, otherwise PSI is mostly sampling noise
    bins = int(np.clip(len(cur) // 5, 3, 10))
    drift = psi(ref.to_numpy(), cur, bins=bins) if len(ref) >= 30 and len(cur) >= 7 else None

    alerts = []
    if recent_mape > mp["mape_alert_ratio"] * bt_mape:
        alerts.append(f"Rolling {w}-day MAPE {recent_mape:.2f}% > "
                      f"{mp['mape_alert_ratio']}x backtest MAPE {bt_mape:.2f}%")
    if drift is not None and drift > mp["psi_alert"]:
        alerts.append(f"Temperature PSI {drift:.3f} > {mp['psi_alert']} (input drift)")

    df["rolling_mape"] = df["ape"].rolling(w, min_periods=3).mean()
    df[["pred_mw", TARGET, "ape", "rolling_mape"]].round(3).to_csv(
        config.REPORTS_DIR / "monitoring_timeseries.csv", index_label="date")

    report = {
        "status": "alert" if alerts else "ok",
        "alerts": alerts,
        "model": meta["model"],
        "n_evaluated": int(len(df)),
        "overall_mape": round(float(df["ape"].mean()), 3),
        "recent_mape": round(recent_mape, 3),
        "backtest_mape": bt_mape,
        "recent_bias_mw": round(float((recent["pred_mw"] - recent[TARGET]).mean()), 1),
        "temp_psi": None if drift is None else round(drift, 4),
        "window": [str(recent.index.min().date()), str(recent.index.max().date())],
    }
    config.MONITOR_FILE.write_text(json.dumps(report, indent=2))
    log.info("Monitoring: %s", report)
    return report