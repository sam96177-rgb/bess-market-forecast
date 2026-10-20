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
import logging

import numpy as np
import pandas as pd

import src.backtest.engine as engine
from src.backtest.engine import run_backtest
from src.config import load_markets_config
from src.models import registry

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


class _StubQuantileModel:
    """No registered model produces quantiles any more; this keeps the
    engine's multi-quantile output path covered."""

    name = "stub_quantile"
    supports_quantiles = True
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    def fit(self, X, y):
        self.level = float(y.mean())

    def predict(self, X, quantiles=None):
        return pd.DataFrame({str(q): self.level + q for q in quantiles}, index=X.index)


def test_row_count_matches_analytical_expectation_for_quantile_model(monkeypatch):
    monkeypatch.setitem(registry.MODEL_REGISTRY, "stub_quantile", _StubQuantileModel)
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]

    backtest_start = dt.date(2024, 4, 20)
    backtest_end = dt.date(2024, 4, 22)  # 3 delivery days
    results, _ = run_backtest(
        market, "stub_quantile", target_series, weather=None,
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


def test_first_train_days_recorded_and_warns_below_threshold(caplog):
    """PLAN_V5_REVISION.md V5-2 guard: first_train_days is always in the
    metadata, and a loud warning fires only when it's short of the
    configured minimum."""
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]
    backtest_start = target_series.index.min().date() + dt.timedelta(days=50)

    with caplog.at_level(logging.WARNING, logger="src.backtest.engine"):
        _, metadata = run_backtest(
            market, "seasonal_naive", target_series, weather=None,
            backtest_start=backtest_start, backtest_end=backtest_start + dt.timedelta(days=1),
            retrain_every_days=91, min_training_history_days=365,
        )
    assert metadata["first_train_days"] == 50
    assert any("first training window" in r.message for r in caplog.records)


def test_first_train_days_no_warning_when_guard_is_none_or_met(caplog):
    target_series = _synthetic_day_ahead_series(n_days=120)
    market = MARKETS["day_ahead"]
    backtest_start = target_series.index.min().date() + dt.timedelta(days=50)

    with caplog.at_level(logging.WARNING, logger="src.backtest.engine"):
        _, metadata = run_backtest(
            market, "seasonal_naive", target_series, weather=None,
            backtest_start=backtest_start, backtest_end=backtest_start + dt.timedelta(days=1),
            retrain_every_days=91, min_training_history_days=None,
        )
    assert metadata["first_train_days"] == 50
    assert not any("first training window" in r.message for r in caplog.records)


def test_feature_cache_does_not_change_results():
    """PLAN_V5_REVISION.md V5-4: sharing a feature_cache across two
    run_backtest calls for the same market must not change either call's
    results -- a cutoff's cached feature frame is identical to a freshly
    built one, since it depends only on the cutoff and the market."""
    target_series = _synthetic_day_ahead_series(n_days=200)
    market = MARKETS["day_ahead"]
    kwargs = dict(
        market=market, target_series=target_series, weather=None,
        backtest_start=dt.date(2024, 4, 20), backtest_end=dt.date(2024, 4, 26),
        retrain_every_days=91,
    )

    uncached, _ = run_backtest(model_name="seasonal_naive", **kwargs)
    shared_cache: engine.FeatureCache = {}
    with_cache_a, _ = run_backtest(model_name="seasonal_naive", feature_cache=shared_cache, **kwargs)
    with_cache_b, _ = run_backtest(model_name="linear_lags", feature_cache=shared_cache, **kwargs)

    pd.testing.assert_frame_equal(
        uncached.drop(columns="run_id"), with_cache_a.drop(columns="run_id")
    )
    # linear_lags is a different model but the same market/window -- its
    # y_pred can differ from seasonal_naive's, but it must still have used
    # the exact same (cached) feature rows, so row count/targets match.
    assert len(with_cache_b) == len(with_cache_a)
    assert set(with_cache_b["target_time"]) == set(with_cache_a["target_time"])


def test_feature_cache_is_actually_reused_across_models(monkeypatch):
    """A cutoff shared by two models' runs (same market/window) must only
    call build_feature_frame once when a cache is shared between them."""
    target_series = _synthetic_day_ahead_series(n_days=200)
    market = MARKETS["day_ahead"]
    kwargs = dict(
        market=market, target_series=target_series, weather=None,
        backtest_start=dt.date(2024, 4, 20), backtest_end=dt.date(2024, 4, 21),
        retrain_every_days=91,
    )

    call_count = {"n": 0}
    real_build = engine.build_feature_frame

    def counting_build(*args, **kw):
        call_count["n"] += 1
        return real_build(*args, **kw)

    monkeypatch.setattr(engine, "build_feature_frame", counting_build)

    shared_cache: engine.FeatureCache = {}
    run_backtest(model_name="seasonal_naive", feature_cache=shared_cache, **kwargs)
    calls_after_first = call_count["n"]
    assert calls_after_first > 0

    run_backtest(model_name="linear_lags", feature_cache=shared_cache, **kwargs)
    assert call_count["n"] == calls_after_first, (
        "second model's run should have hit the shared cache for every cutoff, "
        "making zero additional build_feature_frame calls"
    )


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
