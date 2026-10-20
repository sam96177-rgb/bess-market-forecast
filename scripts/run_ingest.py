"""CLI: ingest all sources (ENTSO-E, regelleistung, weather) for a date
range, populating data/raw/. Safe to re-run — every source caches by
calendar month and skips what's already downloaded.

Usage:
    python scripts/run_ingest.py
    python scripts/run_ingest.py --start 2024-01-01 --end 2024-12-31

Defaults to config/study.yaml's history_start through its latest
test-window end (PLAN_V5_REVISION.md V5-6) -- a real, possibly multi-hour
operation against three live external APIs, but the study is the one
place that range is defined, so a bare `make ingest` should already cover
it. Pass --start/--end to override for an ad-hoc range.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.data import load_target_series, load_weather, load_wind_and_solar_actual, load_wind_and_solar_forecast  # noqa: E402
from src.config import load_study_config  # noqa: E402

MARKETS = ["day_ahead", "fcr", "afrr_pos", "afrr_neg"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, default: study.yaml history_start")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, default: study.yaml's latest test-window end")
    args = parser.parse_args()

    study = load_study_config()
    end = dt.date.fromisoformat(args.end) if args.end else study.max_test_window_end()
    start = dt.date.fromisoformat(args.start) if args.start else study.history_start

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(f"Ingesting {start} .. {end}\n")

    for market in MARKETS:
        series = load_target_series(market, start, end)
        if len(series):
            print(f"  {market:12s} {len(series):6d} rows  [{series.index.min()} .. {series.index.max()}]")
        else:
            print(f"  {market:12s} 0 rows — check logs above for ingestion failures")

    weather = load_weather(start, end)
    print(f"  {'weather':12s} {len(weather):6d} rows")

    # PLAN_V6_REVISION.md V6-3: both backfilled from history_start, same as
    # every other source above -- only 2024-01 was cached before this.
    wind_solar_forecast = load_wind_and_solar_forecast(start, end)
    print(f"  {'gen_fcst':12s} {len(wind_solar_forecast):6d} rows")
    wind_solar_actual = load_wind_and_solar_actual(start, end)
    print(f"  {'gen_actual':12s} {len(wind_solar_actual):6d} rows")


if __name__ == "__main__":
    main()
