"""Expands FCR/aFRR backtest results from their native 4h block
resolution onto day-ahead's native 15-minute grid, by duplicating each
block's prediction/actual across its sub-intervals.

This is an OUTPUT-ONLY reshaping step, for combining all four markets
onto one common time axis (e.g. a downstream dispatch/optimisation
dataset). It must never feed back into evaluation or training:
duplicating one 4h value into 16 rows adds no information, and scoring
against the duplicated rows would inflate n and distort every metric in
src.evaluation.metrics. src.backtest.engine's results/ files (native
resolution) stay the only input to evaluate_results — models are still
built and scored at native hourly/4h resolution, since that's the
resolution they actually have skill at (see src/features/build.py).
"""
from __future__ import annotations

import pandas as pd

from src.config import MarketConfig

FIFTEEN_MIN = pd.Timedelta(minutes=15)


def expand_results_to_15min(results: pd.DataFrame, market: MarketConfig) -> pd.DataFrame:
    """Duplicate every row of a single-market results frame across its
    native block's 15-minute sub-intervals, so it can be concatenated
    with day-ahead's results on a shared 15-min target_time grid.
    day_ahead itself is already native 15-min and is returned unchanged.
    """
    n_steps = round(market.product_length_hours / 0.25)
    if n_steps <= 1:
        return results.copy()

    offsets = [i * FIFTEEN_MIN for i in range(n_steps)]
    expanded = pd.concat(
        [results.assign(target_time=results["target_time"] + offset) for offset in offsets],
        ignore_index=True,
    )
    return expanded.sort_values(["cutoff_time", "target_time", "quantile"]).reset_index(drop=True)
