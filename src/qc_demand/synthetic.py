"""Realistic synthetic data so tests and CI run without network access."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_daily(start: str = "2019-01-01", end: str = "2023-12-31", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D", name="date")
    doy = idx.dayofyear.to_numpy()
    # Québec-like climate: ~ -10 °C in January, ~ 21 °C in July
    temp = 5.5 - 15.5 * np.cos(2 * np.pi * (doy - 20) / 365.25) + rng.normal(0, 4, len(idx))
    hdd = np.clip(18 - temp, 0, None)
    cdd = np.clip(temp - 18, 0, None)
    weekend = (idx.dayofweek >= 5).astype(float)
    noise = np.zeros(len(idx))
    for t in range(1, len(idx)):  # AR(1) residual
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 250)
    demand = 17_000 + 420 * hdd + 150 * cdd - 900 * weekend + noise
    df = pd.DataFrame({
        "demand_mw": demand,
        "peak_mw": demand * 1.12,
        "temp_mean": temp,
        "temp_max": temp + 4,
        "temp_min": temp - 4,
        "imputed": 0,
    }, index=idx)
    return df
