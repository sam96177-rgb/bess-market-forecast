"""Fast, synthetic-data tests for the walk-forward backtest engine.
Includes regression tests for three bugs caught only by actually running
the engine against real data: (1) horizon_step/y_true silently going
all-NaN because tz-aware timestamps got stripped via .values before a
dict lookup, (2) window slicing by cutoff_time's own date (instead of the
delivery day it targets) silently dropping the first day of every window,
and (3) day_ahead's product is 96 x 15-minute blocks/day, not 24 hourly
ones — real since the EU's SDAC 15-minute MTU reform (live 2025-10-01);
config/markets.yaml and this fixture both reflect that.
"""
import datetime as dt

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.config import load_markets_config

MARKETS = load_markets_config()
STEPS_PER_DAY = 96  # day_ahead's current 15-minute product: 24h / 0.25h


def _synthetic_day_ahead_series(n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * STEPS_PER_DAY, freq="15min", tz="Europe/Berlin")
    hour = idx.hour.values
    weekday = idx.dayofweek.values
    values = 40 + 15 * np.sin(hour / 24 * 2 * np.pi) + 5 * (weekday >= 5) + rng.normal(0, 2, len(idx))
    return pd.Series(values, index=idx)


def test_row_count_matches_analytical_expectation_for_point_model():
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]

    backtest_start = dt.date(2024, 4, 20)
    backtest_end = dt.date(2024, 4, 26)  # 7 delivery days
    results, metadata = run_backtest(
        market, "lgbm_point", target_series, weather=None,
        backtest_start=backtest_start, backtest_end=backtest_end,
        retrain_every_days=91,
    )
    expected_rows = 7 * STEPS_PER_DAY * 1  # 7 days, 96 quarter-hourly targets/day, 1 quantile column ("0.5")
    assert len(results) == expected_rows
    assert metadata["n_windows_trained"] == 1
    assert metadata["n_windows_skipped"] == 0


def test_row_count_matches_analytical_expectation_for_quantile_model():
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]

    backtest_start = dt.date(2024, 4, 20)
    backtest_end = dt.date(2024, 4, 22)  # 3 delivery days
    results, _ = run_backtest(
        market, "lgbm_quantile", target_series, weather=None,
        backtest_start=backtest_start, backtest_end=backtest_end,
        retrain_every_days=91,
    )
    expected_rows = 3 * STEPS_PER_DAY * 7  # 3 days, 96 quarter-hours, 7 quantiles
    assert len(results) == expected_rows


def test_horizon_step_and_y_true_are_never_all_null():
    """Regression test: .values on a tz-aware Series silently drops tzinfo,
    which broke every downstream dict-lookup (horizon_step, y_true) into
    producing NaN without raising. Both must be fully populated."""
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]
    results, _ = run_backtest(
        market, "lgbm_point", target_series, weather=None,
        backtest_start=dt.date(2024, 4, 20), backtest_end=dt.date(2024, 4, 22),
        retrain_every_days=91,
    )
    assert results["horizon_step"].notna().all()
    assert results["y_true"].notna().all()
    assert set(results["horizon_step"].unique()) == set(range(1, STEPS_PER_DAY + 1))


def test_window_slicing_does_not_drop_the_first_delivery_day():
    """Regression test: window_cutoffs was filtered by cutoff_time's own
    date, which is D-1 relative to the delivery day it targets — silently
    excluding the window's first requested day. Every requested delivery
    day must produce exactly 96 quarter-hourly target rows (per quantile)."""
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]
    backtest_start, backtest_end = dt.date(2024, 4, 20), dt.date(2024, 4, 26)
    results, _ = run_backtest(
        market, "lgbm_point", target_series, weather=None,
        backtest_start=backtest_start, backtest_end=backtest_end,
        retrain_every_days=91,
    )
    delivery_days = results["target_time"].dt.date.unique()
    expected_days = {backtest_start + dt.timedelta(days=i) for i in range(7)}
    assert set(delivery_days) == expected_days
    counts_per_day = results.groupby(results["target_time"].dt.date).size()
    assert (counts_per_day == STEPS_PER_DAY).all()


def test_insufficient_history_is_skipped_not_crashed():
    """Requesting a backtest starting on the series' very first available
    day leaves zero prior days to train on — the engine should skip that
    window and record it in metadata, not raise."""
    target_series = _synthetic_day_ahead_series(n_days=5)
    market = MARKETS["day_ahead"]
    results, metadata = run_backtest(
        market, "lgbm_point", target_series, weather=None,
        backtest_start=target_series.index.min().date(),
        backtest_end=target_series.index.min().date() + dt.timedelta(days=1),
        retrain_every_days=91,
    )
    assert metadata["n_windows_skipped"] >= 1
    assert results.empty


def test_metadata_has_required_provenance_fields():
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]
    _, metadata = run_backtest(
        market, "seasonal_naive", target_series, weather=None,
        backtest_start=dt.date(2024, 4, 20), backtest_end=dt.date(2024, 4, 21),
        retrain_every_days=91,
    )
    for key in ["run_id", "market", "model", "quantiles", "kept_features", "dropped_features", "data_version", "git_commit", "created_at"]:
        assert key in metadata


def test_results_schema_matches_milestone_5_contract():
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]
    results, _ = run_backtest(
        market, "seasonal_naive", target_series, weather=None,
        backtest_start=dt.date(2024, 4, 20), backtest_end=dt.date(2024, 4, 21),
        retrain_every_days=91,
    )
    assert list(results.columns) == [
        "market", "model", "run_id", "cutoff_time", "target_time",
        "horizon_step", "quantile", "y_pred", "y_true",
    ]
    assert (results["quantile"] == 0.5).all()  # point model
