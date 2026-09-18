"""Evaluation metrics for the long-format backtest results table
(src.backtest.engine's schema: market, model, run_id, cutoff_time,
target_time, horizon_step, quantile, y_pred, y_true).

Point metrics: MAE, RMSE, and a skill score vs. seasonal_naive in ratio
form (1 - rmse_model / rmse_baseline) — a ratio is comparable across
markets with very different price levels (EUR/MWh day-ahead vs. EUR/MW
capacity), where raw RMSE is not.

Probabilistic metrics: pinball loss per quantile, empirical coverage of
the 50% and 80% intervals, Winkler (interval) score, and interval width.
Coverage is the headline number here — a model whose 80% interval covers
only 55% of outcomes is dangerous for a bidding decision no matter how
good its RMSE looks, which is exactly the failure mode these metrics
exist to catch.

Everything is computable sliced by any combination of: market, model,
horizon_step, hour_of_day, month, is_weekend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COVERAGE_LEVELS = {
    50: (0.25, 0.75),
    80: (0.1, 0.9),
}
ALL_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def pivot_to_wide(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (market, model, cutoff_time, target_time), one column
    per quantile level (q_<level>), plus y_true, horizon_step, and derived
    slice dimensions (hour_of_day, month, is_weekend) that aren't in the
    long-format table directly.
    """
    idx_cols = ["market", "model", "run_id", "cutoff_time", "target_time", "horizon_step", "y_true"]
    wide = results.pivot_table(index=idx_cols, columns="quantile", values="y_pred").reset_index()
    wide = wide.rename(columns={q: f"q_{q}" for q in wide.columns if isinstance(q, float)})
    wide.columns.name = None

    target_time = pd.to_datetime(wide["target_time"])
    wide["hour_of_day"] = target_time.dt.hour
    wide["month"] = target_time.dt.month
    wide["is_weekend"] = target_time.dt.dayofweek >= 5
    return wide


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Standard interval (Winkler) score: interval width, plus a penalty
    proportional to 2/alpha for every point that falls outside it."""
    width = upper - lower
    score = width.copy()
    below = y_true < lower
    above = y_true > upper
    score = np.where(below, width + (2 / alpha) * (lower - y_true), score)
    score = np.where(above, width + (2 / alpha) * (y_true - upper), score)
    return float(np.mean(score))


def _has_full_quantiles(g: pd.DataFrame) -> bool:
    return all(f"q_{q}" in g.columns and g[f"q_{q}"].notna().any() for q in ALL_QUANTILES)


def evaluate(wide: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregates the wide-format frame (see pivot_to_wide) into one row
    per group_cols combination, with point metrics always present and
    probabilistic metrics present only for groups where every quantile in
    ALL_QUANTILES was actually produced (point-only models just won't have
    those columns — not an error, just not applicable)."""
    rows = []
    for keys, g in wide.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(g)

        if "q_0.5" in g.columns:
            err = g["q_0.5"].values - g["y_true"].values
            row["mae"] = float(np.mean(np.abs(err)))
            row["rmse"] = float(np.sqrt(np.mean(err**2)))

        if _has_full_quantiles(g):
            y_true = g["y_true"].values
            for q in ALL_QUANTILES:
                row[f"pinball_q{q}"] = pinball_loss(y_true, g[f"q_{q}"].values, q)
            for level, (lo_q, hi_q) in COVERAGE_LEVELS.items():
                lower, upper = g[f"q_{lo_q}"].values, g[f"q_{hi_q}"].values
                row[f"coverage_{level}"] = interval_coverage(y_true, lower, upper)
                row[f"winkler_{level}"] = winkler_score(y_true, lower, upper, alpha=1 - level / 100)
                row[f"interval_width_{level}"] = float(np.mean(upper - lower))

        rows.append(row)
    return pd.DataFrame(rows)


def add_skill_score(metrics: pd.DataFrame, group_cols: list[str], baseline: str = "seasonal_naive") -> pd.DataFrame:
    """Adds skill_score_vs_naive = 1 - rmse / baseline_rmse, matched
    within each slice that isn't "model" (so day_ahead's skill score is
    computed against day_ahead's own seasonal_naive on the same horizon/
    hour/month/weekend slice, never mixed across slices)."""
    if "model" not in group_cols or "rmse" not in metrics.columns:
        return metrics

    slice_cols = [c for c in group_cols if c != "model"]
    baseline_rows = metrics[metrics["model"] == baseline]

    if slice_cols:
        baseline_rmse = baseline_rows.set_index(slice_cols)["rmse"]

        def lookup(row):
            key = tuple(row[c] for c in slice_cols) if len(slice_cols) > 1 else row[slice_cols[0]]
            return baseline_rmse.get(key, np.nan)
    else:
        baseline_scalar = baseline_rows["rmse"].iloc[0] if len(baseline_rows) else np.nan

        def lookup(row):
            return baseline_scalar

    metrics = metrics.copy()
    metrics["baseline_rmse"] = metrics.apply(lookup, axis=1)
    metrics["skill_score_vs_naive"] = 1 - metrics["rmse"] / metrics["baseline_rmse"]
    return metrics


def evaluate_results(
    results: pd.DataFrame, group_by: list[str], baseline: str = "seasonal_naive"
) -> pd.DataFrame:
    """Convenience entry point: pivot + evaluate + skill score in one call."""
    wide = pivot_to_wide(results)
    metrics = evaluate(wide, group_by)
    return add_skill_score(metrics, group_by, baseline=baseline)
