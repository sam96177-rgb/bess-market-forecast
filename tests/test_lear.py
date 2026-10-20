"""PLAN_V8_REVISION.md V8-4: the price-history feature block and the LEAR model."""
import logging

import numpy as np
import pandas as pd
import pytest

from src.backtest.cutoffs import generate_cutoffs
from src.config import load_markets_config
from src.features.build import LEAR_PREFIX, PRICE_HISTORY_DAYS, build_feature_frame
from src.models.base import select_feature_columns
from src.models.lear import LEAR

MARKETS = load_markets_config()
TZ = "Europe/Berlin"


def _quarter_series(start: str, days: int, seed: int = 0) -> pd.Series:
    index = pd.date_range(start, periods=days * 96, freq="15min", tz=TZ)
    return pd.Series(np.random.default_rng(seed).uniform(20, 120, len(index)), index=index)


def _day_ahead_features(series: pd.Series, delivery_day: str, cutoff: str | None = None):
    market = MARKETS["day_ahead"]
    targets = pd.Series(pd.date_range(delivery_day, periods=96, freq="15min", tz=TZ))
    cutoff_time = pd.Timestamp(cutoff, tz=TZ) if cutoff else pd.Timestamp(delivery_day, tz=TZ) - pd.Timedelta(hours=13)
    return build_feature_frame(market, cutoff_time, targets, series)


def test_day_ahead_gets_hourly_mean_prices_of_the_four_past_days():
    series = _quarter_series("2026-01-01", 12)
    frame, dropped = _day_ahead_features(series, "2026-01-10")
    price_cols = [c for c in frame.columns if c.startswith(LEAR_PREFIX)]
    assert len(price_cols) == len(PRICE_HISTORY_DAYS) * 24
    assert not [c for c in dropped if c.startswith(LEAR_PREFIX)]
    # d3 = 7 Jan, hour 5: the mean of its four quarter-hours
    expected = series["2026-01-07 05:00":"2026-01-07 05:45"].mean()
    assert frame[f"{LEAR_PREFIX}d3_s05"].iloc[0] == pytest.approx(expected)
    # same profile on every row of the delivery day
    assert (frame[price_cols].nunique() == 1).all()


def test_blocks_markets_get_six_block_prices_per_past_day():
    market = MARKETS["fcr"]
    index = pd.date_range("2026-01-01", periods=12 * 6, freq="4h", tz=TZ)
    series = pd.Series(np.arange(len(index), dtype=float), index=index)
    targets = pd.Series(pd.date_range("2026-01-10", periods=6, freq="4h", tz=TZ))
    frame, dropped = build_feature_frame(market, pd.Timestamp("2026-01-09 07:00", tz=TZ), targets, series)
    assert len([c for c in frame.columns if c.startswith(LEAR_PREFIX)]) == len(PRICE_HISTORY_DAYS) * 6
    assert not [c for c in dropped if c.startswith(LEAR_PREFIX)]
    assert frame[f"{LEAR_PREFIX}d1_s02"].iloc[0] == series["2026-01-09 08:00"]


@pytest.mark.parametrize("market_name", ["day_ahead", "fcr", "afrr_pos", "afrr_neg"])
def test_every_price_column_passes_the_availability_filter_at_the_real_cutoffs(market_name):
    market = MARKETS[market_name]
    step = "15min" if market.product_length_hours < 1 else f"{int(market.product_length_hours)}h"
    index = pd.date_range("2026-01-01", periods=12 * 24 * (4 if step == "15min" else 1) // (1 if step == "15min" else int(step[:-1])), freq=step, tz=TZ)
    series = pd.Series(1.0, index=index)
    cutoffs = generate_cutoffs(market, pd.Timestamp("2026-01-10").date(), pd.Timestamp("2026-01-10").date())
    cutoffs = cutoffs[cutoffs["horizon_hours"] == min(market.horizons_hours)]
    for cutoff_time, group in cutoffs.groupby("cutoff_time"):
        frame, dropped = build_feature_frame(market, cutoff_time, group["target_window_start"], series)
        assert not [c for c in dropped if c.startswith(LEAR_PREFIX)], market_name
        assert any(c.startswith(LEAR_PREFIX) for c in frame.columns)


def test_a_cutoff_before_yesterdays_prices_are_published_drops_them():
    series = _quarter_series("2026-01-01", 12)
    # delivery 10 Jan; day-ahead prices of 9 Jan are published on 8 Jan, so a cutoff on 7 Jan must drop d1
    _, dropped = _day_ahead_features(series, "2026-01-10", cutoff="2026-01-07 00:00")
    assert f"{LEAR_PREFIX}d1_s00" in dropped


def test_no_price_column_contains_a_price_from_the_delivery_day_or_later():
    series = _quarter_series("2026-01-01", 14)
    poisoned = series.copy()
    poisoned[poisoned.index >= pd.Timestamp("2026-01-10", tz=TZ)] = 1e9
    frame, _ = _day_ahead_features(poisoned, "2026-01-10")
    assert (frame[[c for c in frame.columns if c.startswith(LEAR_PREFIX)]] < 1e6).all().all()


def test_a_missing_hour_stays_blank_never_filled():
    series = _quarter_series("2026-01-01", 12)
    series = series.drop(series["2026-01-09 03:00":"2026-01-09 03:45"].index)
    frame, _ = _day_ahead_features(series, "2026-01-10")
    assert np.isnan(frame[f"{LEAR_PREFIX}d1_s03"].iloc[0]) and frame[f"{LEAR_PREFIX}d1_s04"].notna().all()


def test_the_other_models_never_see_the_price_history_block():
    series = _quarter_series("2026-01-01", 12)
    frame, _ = _day_ahead_features(series, "2026-01-10")
    assert not any(c.startswith(LEAR_PREFIX) for c in select_feature_columns(frame))
    assert "lag_168h" in select_feature_columns(frame)


def _training_frame(n_days: int = 90):
    """A day-ahead-like frame where each period's price is a multiple of yesterday's same hour."""
    series = _quarter_series("2026-01-01", n_days + 10, seed=3)
    frames, ys = [], []
    for d in range(8, n_days + 8):
        day = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        frame, _ = _day_ahead_features(series, day)
        frames.append(frame)
    X = pd.concat(frames).reset_index(drop=True)
    y = X["target_time"].map(series.to_dict())
    return X, y, series


def test_lear_learns_from_the_price_history_and_reports_its_weights():
    X, y, _ = _training_frame()
    model = LEAR()
    model.fit(X.iloc[:-96], y.iloc[:-96])
    preds = model.predict(X.iloc[-96:])
    assert list(preds.columns) == ["0.5"] and len(preds) == 96 and preds["0.5"].notna().all()
    info = model.diagnostics()
    assert info["n_period_models"] == 96 and info["n_inputs"] >= len(PRICE_HISTORY_DAYS) * 24 + 7
    assert 0 <= info["mean_nonzero_weights"] <= info["n_inputs"]


def test_lear_recovers_a_planted_relationship():
    X, y, _ = _training_frame()
    # plant: price at hour h = 2 x yesterday's price at hour h (+ nothing else)
    hour = pd.DatetimeIndex(X["target_time"]).hour
    yesterday = np.array([X.loc[i, f"{LEAR_PREFIX}d1_s{h:02d}"] for i, h in enumerate(hour)])
    planted = 2.0 * yesterday
    model = LEAR()
    model.fit(X.iloc[:-96], pd.Series(planted, index=X.index).iloc[:-96])
    pred = model.predict(X.iloc[-96:])["0.5"].to_numpy()
    assert np.corrcoef(pred, planted[-96:])[0, 1] > 0.9


def test_lear_scores_a_row_with_a_missing_price_slot_and_says_so(caplog):
    X, y, _ = _training_frame(60)
    model = LEAR()
    model.fit(X.iloc[:-96], y.iloc[:-96])
    test = X.iloc[-96:].copy()
    test[f"{LEAR_PREFIX}d1_s02"] = np.nan
    with caplog.at_level(logging.WARNING):
        preds = model.predict(test)
    assert preds["0.5"].notna().all() and "replaced by the training mean" in caplog.text


def test_lear_without_price_history_columns_fails_loudly():
    X = pd.DataFrame({"target_time": pd.date_range("2026-01-01", periods=30, freq="h", tz=TZ), "day_of_week": 3})
    with pytest.raises(ValueError, match="lear_p_"):
        LEAR().fit(X, pd.Series(1.0, index=X.index))


def test_lear_cannot_explode_when_an_input_that_was_nearly_constant_in_training_moves():
    """Solar at 20:45 is ~0 all winter, then nonzero in spring: the standardised value is enormous and
    the asinh back-transform used to turn that into ~1e198 (V8 day-ahead run, 89 days at 20:45)."""
    X, y, _ = _training_frame(60)
    X["solar_forecast_mw"] = 0.0
    X.loc[X.index[:3], "solar_forecast_mw"] = 0.01
    price = X[f"{LEAR_PREFIX}d1_s20"].to_numpy()
    y = pd.Series(price + 5.0 * (X["solar_forecast_mw"] > 0), index=X.index)  # make the tiny input look useful
    model = LEAR()
    model.fit(X.iloc[:-96], y.iloc[:-96])
    test = X.iloc[-96:].copy()
    test["solar_forecast_mw"] = 5000.0
    preds = model.predict(test)["0.5"]
    assert preds.abs().max() < 10 * y.abs().max()
