"""Download and clean Hydro-Québec demand + Open-Meteo weather into one daily table.

Sources
- Hydro-Québec, "Historique de la demande d'électricité au Québec" (hourly, one XLSX
  per year, columns "Date" and "Moyenne (MW)", dates as dd/mm/yyyy hh:mm). Licence CC-BY-NC 4.0.
  Values are hour-ending: 2019-01-01 02:00 is the average of 01:05-02:00.
- Open-Meteo historical archive (ERA5 reanalysis), no API key, CC BY 4.0.
"""
from __future__ import annotations

import logging
import re
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


DATE_LIKE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})")


def _as_datetimes(col: pd.Series) -> pd.Series:
    """Datetime cells and date-like strings only; plain numbers are not dates."""
    is_dt = col.map(lambda v: isinstance(v, pd.Timestamp) or
                    (hasattr(v, "year") and not isinstance(v, str)))
    is_str = col.map(lambda v: isinstance(v, str) and bool(DATE_LIKE.match(v)))
    out = pd.Series(pd.NaT, index=col.index, dtype="datetime64[ns]")
    if is_dt.any():
        out[is_dt] = pd.to_datetime(col[is_dt].astype("object"))
    iso = is_str & col.map(lambda v: isinstance(v, str) and v.strip()[:4].isdigit())
    other = is_str & ~iso
    if iso.any():  # 2019-02-05 ...: year first, unambiguous
        out[iso] = pd.to_datetime(col[iso].astype("object"), errors="coerce", format="mixed")
    if other.any():  # HQ files are French: 05/02/2019 is 5 February, not 2 May
        out[other] = pd.to_datetime(col[other].astype("object"), errors="coerce",
                                    format="mixed", dayfirst=True)
    return out


def _as_numbers(col: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(col):  # French decimals: "21 345,6" -> 21345.6
        col = col.map(lambda v: v.replace(" ", "").replace(" ", "").replace(",", ".")
                      if isinstance(v, str) else v)
    return pd.to_numeric(col, errors="coerce")


def _parse_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """Find the timestamp and demand columns by their CONTENT, not their labels.

    Labels vary between files and years (accents, units, extra title rows), so:
    - timestamp column = the column with the most date-like cells;
    - demand column    = the column with the most values in a plausible MW range;
    - if timestamps carry no time of day, an hour column (1-24) is combined with them.
    """
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw.columns = range(raw.shape[1])
    n = len(raw)
    dates = {c: _as_datetimes(raw[c]) for c in raw.columns}
    date_share = {c: s.notna().mean() for c, s in dates.items()}
    date_col = max(date_share, key=date_share.get)
    if date_share[date_col] < 0.5:
        raise ValueError("no date column")
    nums = {c: _as_numbers(raw[c]) for c in raw.columns if c != date_col}
    mw_share = {c: s.between(5_000, 60_000).mean() for c, s in nums.items()}
    if not mw_share or max(mw_share.values()) < 0.5:
        raise ValueError("no demand column in the 5,000-60,000 MW range")
    dem_col = max(mw_share, key=mw_share.get)

    ts = dates[date_col]
    valid = ts.dropna()
    if len(valid) and (valid.dt.hour == 0).all() and (valid.dt.minute == 0).all():
        hour_share = {c: s.between(0, 24).mean() for c, s in nums.items()
                      if c != dem_col and s.dropna().apply(float.is_integer).all()}
        hour_col = max(hour_share, key=hour_share.get) if hour_share else None
        if hour_col is None or hour_share[hour_col] < 0.5:
            raise ValueError("timestamps have no time of day and no hour column was found")
        ts = ts + pd.to_timedelta(nums[hour_col], unit="h")
        log.info("Combined date column %s with hour column %s", date_col, hour_col)

    df = pd.DataFrame({"timestamp": ts, "demand_mw": nums[dem_col]}).dropna(subset=["timestamp"])
    log.info("Parsed %d of %d rows (date col %s, demand col %s)", len(df), n, date_col, dem_col)
    return df


def parse_hq_xlsx(path: Path) -> pd.DataFrame:
    """Return hourly frame [timestamp, demand_mw] from one Hydro-Québec yearly file."""
    sheets = pd.read_excel(path, header=None, sheet_name=None)
    errors = {}
    for name, raw in sheets.items():
        try:
            df = _parse_sheet(raw)
            break
        except ValueError as e:
            errors[name] = str(e)
    else:
        preview = "\n\n".join(f"--- sheet {k!r}: {errors[k]}\n{v.head(8).to_string()}"
                              for k, v in sheets.items())
        raise ValueError(f"Could not find timestamp + demand columns in {path.name}.\n"
                         f"First rows of each sheet:\n{preview}")
    # hour-ending -> hour-beginning so 00:00 belongs to the previous day correctly
    df["timestamp"] = df["timestamp"] - pd.Timedelta(hours=1)
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


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


# ----------------------------------------------------- weather
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


# ---------------------------------------------------------- quality
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