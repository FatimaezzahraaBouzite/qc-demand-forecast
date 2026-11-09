"""AWS Lambda entry point: daily forecast + monitoring, state kept in S3.

S3 layout (bucket from env BUCKET):
  models/model.joblib, models/model_meta.json   <- written by training / retrain workflow
  data/daily.parquet                             <- history (updated by ingest-recent in live)
  forecasts/forecasts.csv                        <- appended every run
  reports/monitoring.json                        <- latest monitoring report

Event {"mode": "replay-next"} (default) forecasts the day after the last logged forecast
using held-out history: a fully working daily pipeline without depending on the live feed.
Event {"mode": "live"} pulls recent demand, then forecasts tomorrow with forecast weather.
"""
import json
import os
import shutil
from pathlib import Path

import boto3

WORK = Path("/tmp/qc")
os.environ["QC_DEMAND_ROOT"] = str(WORK)   # must be set before importing qc_demand

from qc_demand import config, monitor, predict  # noqa: E402

s3 = boto3.client("s3")
BUCKET = os.environ["BUCKET"]
FILES = {
    "models/model.joblib": config.MODEL_FILE,
    "models/model_meta.json": config.MODEL_META,
    "data/daily.parquet": config.DAILY_FILE,
    "forecasts/forecasts.csv": config.FORECAST_LOG,
}


def _pull():
    config.ensure_dirs()
    here = Path(__file__).resolve().parent
    src = next(p for p in (here / "params.yaml", here.parent / "params.yaml") if p.exists())
    shutil.copy(src, WORK / "params.yaml")
    for key, path in FILES.items():
        try:
            s3.download_file(BUCKET, key, str(path))
        except s3.exceptions.ClientError:
            if key.startswith("forecasts/"):
                continue  # first run: no log yet
            raise


def _push():
    for key in ("forecasts/forecasts.csv", "data/daily.parquet"):
        s3.upload_file(str(FILES[key]), BUCKET, key)
    s3.upload_file(str(config.MONITOR_FILE), BUCKET, "reports/monitoring.json")


def _event_mode(event) -> str:
    """Accept the event however it arrives: a dict (schedule, console), a JSON string
    (some CLI/PowerShell invocations double-encode it), or nothing at all."""
    for _ in range(3):  # unwrap up to 3 layers of JSON encoding
        if not isinstance(event, (str, bytes)):
            break
        try:
            event = json.loads(event)
        except ValueError:
            event = {"mode": event.decode() if isinstance(event, bytes) else event}
    if not isinstance(event, dict):
        event = {}
    mode = event.get("mode", "replay-next")
    if mode not in ("replay-next", "live"):
        raise ValueError(f"Unknown mode {mode!r}; use 'replay-next' or 'live'")
    return mode


def handler(event, _context):
    import pandas as pd

    _pull()
    params = config.load_params()
    mode = _event_mode(event)

    if mode == "live":
        predict.ingest_recent(params)
        predict.live(params)
    else:
        meta = json.loads(config.MODEL_META.read_text())
        if config.FORECAST_LOG.exists():
            last = pd.read_csv(config.FORECAST_LOG)["target_date"].max()
        else:
            last = meta["trained_until"]
        nxt = pd.Timestamp(last) + pd.Timedelta(days=1)
        if nxt > pd.read_parquet(config.DAILY_FILE).index.max():
            return {"mode": mode, "status": "replay window exhausted; switch the "
                    "schedule input to {\"mode\": \"live\"}"}
        predict.replay(params, nxt.date().isoformat(), nxt.date().isoformat())

    report = monitor.run(params)
    _push()

    topic = os.environ.get("ALERT_TOPIC_ARN")
    if topic and report.get("status") == "alert":
        boto3.client("sns").publish(TopicArn=topic, Subject="QC demand forecast alert",
                                    Message=json.dumps(report, indent=2))
    return {"mode": mode, "monitoring": report}