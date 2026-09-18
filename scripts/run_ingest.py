"""CLI: ingest all sources (ENTSO-E, regelleistung, weather) for a date
range, populating data/raw/. Safe to re-run — every source caches by
calendar month and skips what's already downloaded.

Usage:
    python scripts/run_ingest.py
    python scripts/run_ingest.py --start 2024-01-01 --end 2024-12-31

Defaults to the most recent 60 days (through yesterday) rather than the
full "2024-01-01 to present" range from the project brief: fetching years
of history is a real, possibly multi-hour operation against three live
external APIs, and shouldn't be what a first `make ingest` silently
kicks off. Pass --start/--end for a full historical backfill.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.data import load_target_series, load_weather  # noqa: E402

MARKETS = ["day_ahead", "fcr", "afrr_pos", "afrr_neg"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, default: 60 days before --end")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, default: yesterday")
    args = parser.parse_args()

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today() - dt.timedelta(days=1)
    start = dt.date.fromisoformat(args.start) if args.start else end - dt.timedelta(days=60)

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


if __name__ == "__main__":
    main()
