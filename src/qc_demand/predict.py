"""Produce day-ahead forecasts and append them to the forecast log.

Two modes
- replay: forecast dates inside the historical table, hiding that day's demand. With a
  model trained on data up to a cutoff (`train --until`), this simulates production on
  data the model never saw, so monitoring has real actuals to compare against.
- live:   forecast tomorrow using Open-Meteo's weather *forecast* for the target day.
          Needs the daily history to run up to yesterday (see ingest_recent).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests

from . import config
from .features import TARGET, build_features
from .ingest import fetch_weather_forecast

log = logging.getLogger(__name__)

HQ_REALTIME_JSON = "https://www.hydroquebec.com/data/documents-donnees/donnees-ouvertes/json/demande.json"


def load_model():
    model = joblib.load(config.MODEL_FILE)
    meta = json.loads(config.MODEL_META.read_text())
    return model, meta


def forecast_day(model, daily: pd.DataFrame, target: pd.Timestamp, weather_row: dict,
                 params: dict) -> float:
    """Forecast one day from history strictly before `target` plus that day's weather."""
    hist = daily.loc[: target - pd.Timedelta(days=1)].copy()
    row = pd.DataFrame({TARGET: np.nan, "peak_mw": np.nan, **weather_row}, index=[target])
    frame = build_features(pd.concat([hist, row]).asfreq("D"), params)
    return float(model.predict(frame, pd.DatetimeIndex([target])).iloc[0])


def _append_log(rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if config.FORECAST_LOG.exists():
        old = pd.read_csv(config.FORECAST_LOG)
        new = pd.concat([old, new]).drop_duplicates(["target_date", "mode"], keep="last")
    new.sort_values("target_date").to_csv(config.FORECAST_LOG, index=False)
    return new


def replay(params: dict, start: str, end: str) -> pd.DataFrame:
    model, meta = load_model()
    daily = pd.read_parquet(config.DAILY_FILE)
    if pd.Timestamp(start) <= pd.Timestamp(meta["trained_until"]):
        log.warning("Replay starts before trained_until=%s: those forecasts are in-sample",
                    meta["trained_until"])
    issued = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for d in pd.date_range(start, end, freq="D"):
        w = daily.loc[d, ["temp_mean", "temp_max", "temp_min"]].to_dict()
        rows.append({"target_date": d.date().isoformat(), "pred_mw": round(
            forecast_day(model, daily, d, w, params), 1), "model": meta["model"],
            "mode": "replay", "issued_at": issued, "temp_mean": round(w["temp_mean"], 2)})
    log.info("Replayed %d forecasts", len(rows))
    return _append_log(rows)


def parse_hq_realtime(payload) -> pd.DataFrame:
    """Best-effort parser for HQ's near-real-time demand JSON (15-min values).

    VERIFY the structure on first use (open the URL in a browser): this walks every
    record and picks the first date-like and first numeric demand-like field.
    """
    records = payload.get("details", payload) if isinstance(payload, dict) else payload
    out = []
    for rec in records:
        flat = {**rec, **rec.get("valeurs", {})} if isinstance(rec, dict) else {}
        ts = next((v for k, v in flat.items() if "date" in k.lower()), None)
        val = next((v for k, v in flat.items()
                    if "demande" in k.lower() and isinstance(v, (int, float))), None)
        if ts is not None and val is not None:
            out.append((pd.to_datetime(ts), float(val)))
    if not out:
        raise ValueError("Could not parse HQ real-time JSON; inspect it and update the parser")
    df = pd.DataFrame(out, columns=["timestamp", "demand_mw"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df


def ingest_recent(params: dict) -> pd.DataFrame:
    """Append recent complete days from the real-time feed to the daily table."""
    from .ingest import fetch_weather_history

    r = requests.get(HQ_REALTIME_JSON, timeout=60)
    r.raise_for_status()
    rt = parse_hq_realtime(r.json())
    rt["date"] = rt["timestamp"].dt.normalize()
    g = rt.groupby("date")["demand_mw"]
    recent = pd.DataFrame({"demand_mw": g.mean(), "peak_mw": g.max(), "n": g.count()})
    recent = recent[recent["n"] >= 80].drop(columns="n")  # complete days (96 x 15 min)
    if recent.empty:
        return pd.read_parquet(config.DAILY_FILE)
    p = params["data"]
    w = fetch_weather_history(str(recent.index.min().date()), str(recent.index.max().date()),
                              p["cities"], p["timezone"])
    recent = recent.join(w).assign(imputed=0)
    daily = pd.read_parquet(config.DAILY_FILE)
    daily = pd.concat([daily[~daily.index.isin(recent.index)], recent]).sort_index().asfreq("D")
    daily.to_parquet(config.DAILY_FILE)
    return daily


def live(params: dict, target: date | None = None) -> pd.DataFrame:
    model, meta = load_model()
    daily = pd.read_parquet(config.DAILY_FILE)
    target = pd.Timestamp(target or (date.today() + timedelta(days=1)))
    if daily[TARGET].last_valid_index() < target - pd.Timedelta(days=1):
        raise RuntimeError(
            f"History ends {daily[TARGET].last_valid_index().date()}; need data through "
            f"{(target - pd.Timedelta(days=1)).date()}. Run `qc-demand ingest-recent` first.")
    wf = fetch_weather_forecast(params["data"]["cities"], params["data"]["timezone"])
    w = wf.loc[target, ["temp_mean", "temp_max", "temp_min"]].to_dict()
    pred = forecast_day(model, daily, target, w, params)
    return _append_log([{"target_date": target.date().isoformat(), "pred_mw": round(pred, 1),
                         "model": meta["model"], "mode": "live",
                         "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "temp_mean": round(w["temp_mean"], 2)}])
