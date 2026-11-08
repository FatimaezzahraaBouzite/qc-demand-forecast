"""Feature engineering for day-ahead forecasting.

Leakage rule: a row for day D only uses demand up to D-1 (lags >= 1) and weather for D
(in production that weather is a forecast, see predict.py).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

TARGET = "demand_mw"


def qc_holidays(years) -> set[pd.Timestamp]:
    h = holidays.Canada(subdiv="QC", years=list(years))
    return {pd.Timestamp(d) for d in h}


def build_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    fp = params["features"]
    out = df.copy()
    idx = out.index

    # calendar
    out["dow"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month
    doy = idx.day_of_year
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    hol = qc_holidays(range(idx.min().year, idx.max().year + 2))
    out["is_holiday"] = idx.isin(list(hol)).astype(int)
    # Christmas-to-New-Year low-activity period
    out["is_xmas_break"] = (((idx.month == 12) & (idx.day >= 24)) |
                            ((idx.month == 1) & (idx.day <= 2))).astype(int)

    # weather (heating dominates in Québec)
    out["hdd"] = (fp["hdd_base_c"] - out["temp_mean"]).clip(lower=0)
    out["cdd"] = (out["temp_mean"] - fp["cdd_base_c"]).clip(lower=0)
    out["temp_range"] = out["temp_max"] - out["temp_min"]
    out["temp_mean_lag1"] = out["temp_mean"].shift(1)
    out["temp_change"] = out["temp_mean"] - out["temp_mean_lag1"]

    # autoregressive (only past demand)
    for lag in fp["lags"]:
        out[f"lag_{lag}"] = out[TARGET].shift(lag)
    past = out[TARGET].shift(1)
    for w in fp["rolling_windows"]:
        out[f"roll_mean_{w}"] = past.rolling(w, min_periods=max(2, w // 2)).mean()
    out["roll_std_7"] = past.rolling(7, min_periods=4).std()
    return out


def feature_columns(params: dict) -> list[str]:
    fp = params["features"]
    cols = ["dow", "is_weekend", "month", "doy_sin", "doy_cos", "is_holiday", "is_xmas_break",
            "temp_mean", "temp_max", "temp_min", "hdd", "cdd", "temp_range",
            "temp_mean_lag1", "temp_change"]
    cols += [f"lag_{lag}" for lag in fp["lags"]]
    cols += [f"roll_mean_{w}" for w in fp["rolling_windows"]] + ["roll_std_7"]
    return cols


def model_ready(feats: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Drop warm-up rows that lack lag history."""
    cols = feature_columns(params)
    return feats.dropna(subset=cols + [TARGET])