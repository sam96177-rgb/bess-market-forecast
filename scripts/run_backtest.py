"""CLI: run the walk-forward backtest for one or more (market, model)
pairs and save results to results/. Requires data/raw/ to already be
populated (run scripts/run_ingest.py first for the same date range).

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2024-01-18 --end 2024-01-31 --markets day_ahead fcr --models seasonal_naive lgbm_quantile

Defaults to the most recent 60 days and every registered model across
every market — see scripts/run_ingest.py's docstring for why the default
window isn't the full historical range.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.data import load_target_series, load_weather  # noqa: E402
from src.backtest.engine import run_backtest, save_run  # noqa: E402
from src.config import load_markets_config  # noqa: E402
from src.models.registry import MODEL_REGISTRY  # noqa: E402

ALL_MARKETS = ["day_ahead", "fcr", "afrr_pos", "afrr_neg"]
HISTORY_BUFFER_DAYS = 30  # extra lookback so lag_672h etc. have real history at the start of the window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, default: 60 days before --end")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, default: yesterday")
    parser.add_argument("--markets", nargs="+", default=ALL_MARKETS, choices=ALL_MARKETS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY))
    parser.add_argument("--n-trials", type=int, default=15, help="Optuna trials for lgbm_quantile_tuned")
    args = parser.parse_args()

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today() - dt.timedelta(days=1)
    start = dt.date.fromisoformat(args.start) if args.start else end - dt.timedelta(days=60)
    ingest_start = start - dt.timedelta(days=HISTORY_BUFFER_DAYS)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.features.availability").setLevel(logging.ERROR)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    markets_cfg = load_markets_config()
    weather = load_weather(ingest_start, end)

    for market_name in args.markets:
        cfg = markets_cfg[market_name]
        target_series = load_target_series(market_name, ingest_start, end)
        for model_name in args.models:
            kwargs = {"n_trials": args.n_trials} if model_name == "lgbm_quantile_tuned" else {}
            print(f"\n=== {market_name} / {model_name} ===")
            results, metadata = run_backtest(cfg, model_name, target_series, weather, start, end, model_kwargs=kwargs)
            if results.empty:
                print("  no rows produced (not enough training history for this window) — skipped")
                continue
            parquet_path, _ = save_run(results, metadata)
            print(f"  rows={len(results)}  windows_trained={metadata['n_windows_trained']}  -> {parquet_path.name}")


if __name__ == "__main__":
    main()
