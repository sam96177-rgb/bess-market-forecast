"""CLI: export a 15-minute-grid version of the backtest results, for
combining all four markets on one common time axis (e.g. a downstream
dispatch/optimisation dataset). FCR/aFRR are duplicated across each 4h
block's sixteen 15-min sub-intervals; day-ahead is already native
15-min and passes through unchanged.

This is an OUTPUT-ONLY reshaping step. It never changes what gets
forecast or how models are scored — src.evaluation.metrics always reads
the native-resolution results/ files directly, because that's the
resolution the models actually have skill at. See
src/evaluation/resample.py for why the duplicated rows must not be fed
back into evaluation.

Usage:
    python scripts/export_15min_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.config import load_markets_config  # noqa: E402
from src.evaluation.resample import expand_results_to_15min  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results_15min"


def main() -> None:
    markets_cfg = load_markets_config()
    paths = sorted(RESULTS_DIR.glob("*.parquet"))
    if not paths:
        print(f"No results found in {RESULTS_DIR} — run scripts/run_backtest.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        results = pd.read_parquet(path)
        if results.empty:
            continue
        market_name = results["market"].iloc[0]
        expanded = expand_results_to_15min(results, markets_cfg[market_name])
        out_path = OUTPUT_DIR / path.name
        expanded.to_parquet(out_path)
        print(f"{path.name}: {market_name}  {len(results)} rows -> {len(expanded)} rows  -> {out_path}")


if __name__ == "__main__":
    main()
