"""CLI: run the walk-forward backtest for one or more (market, model)
pairs and save results to results/. Requires data/raw/ to already be
populated (run scripts/run_ingest.py first for the same date range).

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --start 2024-01-18 --end 2024-01-31 --markets day_ahead fcr --models seasonal_naive lgbm_point

Defaults to config/study.yaml's per-market test_window and models
(PLAN_V5_REVISION.md V5-6). --start/--end, if given, override the study
window uniformly for every requested market -- for ad-hoc runs outside
the study.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.data import load_target_series, load_weather, load_wind_and_solar_actual, load_wind_and_solar_forecast  # noqa: E402
from src.backtest.engine import run_backtest, save_run  # noqa: E402
from src.config import load_entsoe_generation_forecast_published_local, load_markets_config, load_study_config  # noqa: E402
from src.models.registry import MODEL_REGISTRY  # noqa: E402

ALL_MARKETS = ["day_ahead", "fcr", "afrr_pos", "afrr_neg"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, default: study.yaml test_window per market")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, default: study.yaml test_window per market")
    parser.add_argument("--markets", nargs="+", default=ALL_MARKETS, choices=ALL_MARKETS)
    parser.add_argument("--models", nargs="+", default=None, help="default: study.yaml's models", choices=list(MODEL_REGISTRY))
    args = parser.parse_args()

    study = load_study_config()
    models = args.models if args.models is not None else study.models
    override_start = dt.date.fromisoformat(args.start) if args.start else None
    override_end = dt.date.fromisoformat(args.end) if args.end else None
    ingest_start = study.history_start  # V5-2: load from the study's full backfill, not a short rolling buffer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.features.availability").setLevel(logging.ERROR)
    logging.getLogger("lightgbm").setLevel(logging.ERROR)

    markets_cfg = load_markets_config()
    # V6-3: one ENTSO-E-wide assumption, not per-market -- see
    # config/markets.yaml's entsoe_generation_forecast_published_local.
    generation_forecast_published_local = load_entsoe_generation_forecast_published_local()

    for market_name in args.markets:
        window = study.test_window[market_name]
        # V6-4: default start is the validation window's start when configured,
        # so one backtest run forecasts validation + test in one pass; the test
        # window's own end is untouched, and all test-window scoring stays as-is.
        start = override_start or study.forecast_start(market_name)
        end = override_end or window.end
        cfg = markets_cfg[market_name]
        target_series = load_target_series(market_name, ingest_start, end)
        weather = load_weather(ingest_start, end)
        # V6-3: both trained models get wind/solar generation inputs; a lag
        # feature needs history before `start` too, so these are also
        # loaded from ingest_start, same as weather.
        generation_forecast = load_wind_and_solar_forecast(ingest_start, end)
        generation_actual = load_wind_and_solar_actual(ingest_start, end)
        # V5-4: one feature-frame cache per market, shared across every
        # model below -- a cutoff's features don't depend on model_name,
        # so this builds each one once total instead of once per model.
        feature_cache: dict = {}
        for model_name in models:
            print(f"\n=== {market_name} / {model_name} ===")
            results, metadata = run_backtest(
                cfg, model_name, target_series, weather, start, end,
                min_training_history_days=study.min_training_history_days,
                feature_cache=feature_cache,
                generation_forecast=generation_forecast, generation_actual=generation_actual,
                generation_forecast_published_local=generation_forecast_published_local,
            )
            metadata["study"] = study.name
            # V6-3: the assumption recorded on every run, not just the ones that
            # happen to keep a generation feature past the availability filter.
            metadata["generation_forecast_published_local"] = generation_forecast_published_local.isoformat()
            if results.empty:
                print("  no rows produced (not enough training history for this window) — skipped")
                continue
            parquet_path, _ = save_run(results, metadata)
            print(f"  rows={len(results)}  windows_trained={metadata['n_windows_trained']}  -> {parquet_path.name}")


if __name__ == "__main__":
    main()
