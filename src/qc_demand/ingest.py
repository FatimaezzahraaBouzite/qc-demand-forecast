"""Download and clean Hydro-Québec demand + Open-Meteo weather into one daily table.

Sources
- Hydro-Québec, "Historique de la demande d'électricité au Québec" (hourly, one XLSX
  per year, columns "Date" and "Demande (MW)"). Licence CC-BY-NC 4.0.
  Values are hour-ending: 2019-01-01 02:00 is the average of 01:05-02:00.
- Open-Meteo historical archive (ERA5 reanalysis), no API key, CC BY 4.0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS = "temperature_2m_mean,temperature_2m_max,temperature_2m_min"


# --------------------------------------------------------------------------- demand
def download_hq_year(year: int, url_template: str, dest: Path = config.DATA_RAW) -> Path:
    out = dest / f"hq_demand_{year}.xlsx"
    if out.exists():
        return out
    url = url_template.format(year=year)
    log.info("Downloading %s", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def parse_hq_xlsx(path: Path) -> pd.DataFrame:
    """Return hourly frame [timestamp, demand_mw]. Tolerant to header offsets."""
    raw = pd.read_excel(path, header=None)
    # find the header row: the first row containing a cell starting with "Date"
    header_row = next(
        i for i, row in raw.iterrows()
        if any(str(c).strip().lower().startswith("date") for c in row.values)
    )
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row].values]
    date_col = next(c for c in df.columns if c.lower().startswith("date"))
    dem_col = next(c for c in df.columns if "demande" in c.lower())
    df = df[[date_col, dem_col]].rename(columns={date_col: "timestamp", dem_col: "demand_mw"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    # hour-ending -> hour-beginning so 00:00 belongs to the previous day correctly
    df["timestamp"] = df["timestamp"] - pd.Timedelta(hours=1)
    return df


def hourly_to_daily(hourly: pd.DataFrame, min_hours: int = 20) -> pd.DataFrame:
    """Daily mean and peak demand. Days with too few valid hours are set to NaN."""
    h = hourly.copy()
    h["date"] = h["timestamp"].dt.normalize()
    g = h.groupby("date")["demand_mw"]
    daily = pd.DataFrame({
        "demand_mw": g.mean(),
        "peak_mw": g.max(),
        "n_hours": g.count(),
    })
    daily.loc[daily["n_hours"] < min_hours, ["demand_mw", "peak_mw"]] = np.nan
    return daily.drop(columns="n_hours")


# --------------------------------------------------------------------------- weather
def _open_meteo(url: str, lat: float, lon: float, tz: str, **extra) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "daily": DAILY_VARS, "timezone": tz, **extra}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d).rename(columns={
        "time": "date",
        "temperature_2m_mean": "temp_mean",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
    })
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def weighted_weather(frames: dict[str, pd.DataFrame], cities: dict) -> pd.DataFrame:
    total = sum(c["weight"] for c in cities.values())
    out = None
    for name, df in frames.items():
        w = cities[name]["weight"] / total
        out = df * w if out is None else out.add(df * w, fill_value=np.nan)
    return out


def fetch_weather_history(start: str, end: str, cities: dict, tz: str) -> pd.DataFrame:
    frames = {
        name: _open_meteo(OPEN_METEO_ARCHIVE, c["lat"], c["lon"], tz,
                          start_date=start, end_date=end)
        for name, c in cities.items()
    }
    return weighted_weather(frames, cities)


def fetch_weather_forecast(cities: dict, tz: str, days: int = 3) -> pd.DataFrame:
    """Used in production: tomorrow's temperature comes from a forecast, not actuals."""
    frames = {
        name: _open_meteo(OPEN_METEO_FORECAST, c["lat"], c["lon"], tz, forecast_days=days)
        for name, c in cities.items()
    }
    return weighted_weather(frames, cities)


# --------------------------------------------------------------------------- quality
def validate_daily(df: pd.DataFrame) -> dict:
    """Simple data-quality report; raises on hard failures."""
    report = {
        "n_days": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "missing_demand_days": int(df["demand_mw"].isna().sum()),
        "missing_temp_days": int(df["temp_mean"].isna().sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
    }
    if report["duplicate_dates"]:
        raise ValueError(f"Duplicate dates in daily table: {report}")
    dem = df["demand_mw"].dropna()
    if not dem.between(5_000, 50_000).all():
        raise ValueError("Demand outside plausible range 5-50 GW; check parsing/units")
    if report["missing_demand_days"] > 0.05 * len(df):
        raise ValueError(f"More than 5% of days missing demand: {report}")
    return report


def build_daily(params: dict) -> pd.DataFrame:
    config.ensure_dirs()
    p = params["data"]
    hourly = pd.concat(
        [parse_hq_xlsx(download_hq_year(y, p["hq_url_template"])) for y in p["years"]],
        ignore_index=True,
    ).drop_duplicates("timestamp")
    daily = hourly_to_daily(hourly)
    start, end = daily.index.min().date().isoformat(), daily.index.max().date().isoformat()
    weather = fetch_weather_history(start, end, p["cities"], p["timezone"])
    df = daily.join(weather, how="left").asfreq("D")
    df.index.name = "date"
    # short gaps: interpolate demand (max 2 days) so lags stay usable; flag them
    df["imputed"] = df["demand_mw"].isna().astype(int)
    df["demand_mw"] = df["demand_mw"].interpolate(limit=2)
    df["peak_mw"] = df["peak_mw"].interpolate(limit=2)
    return df


def run(params: dict, synthetic: bool = False) -> pd.DataFrame:
    if synthetic:
        from .synthetic import make_daily
        df = make_daily()
    else:
        df = build_daily(params)
    report = validate_daily(df)
    log.info("Data quality: %s", report)
    config.ensure_dirs()
    df.to_parquet(config.DAILY_FILE)
    return df
