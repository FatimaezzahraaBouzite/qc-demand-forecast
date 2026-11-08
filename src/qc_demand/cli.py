"""Command line entry point: `qc-demand <command>` (or `python -m qc_demand.cli`)."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qc-demand")
    ap.add_argument("--params", default=None, help="path to params.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="download + clean demand and weather")
    p.add_argument("--synthetic", action="store_true", help="offline synthetic data")
    sub.add_parser("ingest-recent", help="append recent days from HQ real-time feed")
    p = sub.add_parser("train", help="backtest, log to MLflow, refit best model")
    p.add_argument("--until", default=None, help="last training date, e.g. 2023-06-30")
    p = sub.add_parser("predict", help="day-ahead forecast")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--date", help="live target date (default: tomorrow)")
    p = sub.add_parser("monitor", help="accuracy + drift report")
    p.add_argument("--fail-on-alert", action="store_true")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = config.load_params(a.params)

    if a.cmd == "ingest":
        from .ingest import run
        df = run(params, synthetic=a.synthetic)
        print(f"daily table: {len(df)} days, {df.index.min().date()} -> {df.index.max().date()}")
    elif a.cmd == "ingest-recent":
        from .predict import ingest_recent
        ingest_recent(params)
    elif a.cmd == "train":
        from .train import run
        print(json.dumps(run(params, until=a.until), indent=2))
    elif a.cmd == "predict":
        from . import predict
        if a.mode == "replay":
            if not (a.start and a.end):
                ap.error("replay needs --start and --end")
            out = predict.replay(params, a.start, a.end)
        else:
            out = predict.live(params, a.date)
        print(out.tail(5).to_string(index=False))
    elif a.cmd == "monitor":
        from .monitor import run
        report = run(params)
        print(json.dumps(report, indent=2))
        if a.fail_on_alert and report.get("status") == "alert":
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
