"""PLAN_V6_REVISION.md V6-3: wind/solar generation features --
add_generation_features and build_feature_frame's wiring of them. The
availability math itself (assumed vs. verified publication time, the
lagged-actual anchor) is covered in tests/test_feature_availability.py;
this file covers the join, naming, and registration side.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import load_markets_config
from src.features.availability import FEATURE_AVAILABILITY
from src.features.build import GENERATION_ACTUAL_LAG_HOURS, add_generation_features, build_feature_frame

MARKETS = load_markets_config()
PUBLISHED_AT_07 = dt.time(7, 0)


def _generation_frame(start: str, periods: int, offset_mw: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="Europe/Berlin")
    return pd.DataFrame(
        {"Wind Onshore": np.arange(periods, dtype=float) + offset_mw, "Solar": np.full(periods, 10.0) + offset_mw},
        index=idx,
    )


def test_forecast_columns_are_block_mean_joined_and_renamed():
    market = MARKETS["day_ahead"]  # 0.25h blocks -- reduces the join to a lookup
    target_times = pd.Series(pd.date_range("2026-01-02 00:00", periods=4, freq="1h", tz="Europe/Berlin"))
    forecast = _generation_frame("2026-01-01 00:00", 48)

    out = add_generation_features(target_times, market, forecast, None, PUBLISHED_AT_07)

    assert list(out.columns) == ["wind_onshore_forecast_mw", "solar_forecast_mw"]
    assert out.index.equals(target_times.index)
    # "Wind Offshore" was never in the input frame, so no column for it (no KeyError either).
    assert "wind_offshore_forecast_mw" not in out.columns


def test_actual_columns_are_lagged_by_48h():
    market = MARKETS["day_ahead"]
    target_times = pd.Series([pd.Timestamp("2026-01-03 10:00", tz="Europe/Berlin")])
    actual = _generation_frame("2026-01-01 00:00", 72)

    out = add_generation_features(target_times, market, None, actual, None)

    lag_col = f"wind_onshore_actual_lag{GENERATION_ACTUAL_LAG_HOURS}h_mw"
    assert lag_col in out.columns
    expected_time = target_times.iloc[0] - pd.Timedelta(hours=GENERATION_ACTUAL_LAG_HOURS)
    assert out[lag_col].iloc[0] == pytest.approx(actual.loc[expected_time, "Wind Onshore"])


def test_forecast_without_a_published_time_raises():
    market = MARKETS["day_ahead"]
    target_times = pd.Series([pd.Timestamp("2026-01-02 00:00", tz="Europe/Berlin")])
    forecast = _generation_frame("2026-01-01 00:00", 48)
    with pytest.raises(ValueError, match="forecast_published_local"):
        add_generation_features(target_times, market, forecast, None, None)


def test_registers_availability_rules_for_every_column_it_produces():
    market = MARKETS["day_ahead"]
    target_times = pd.Series([pd.Timestamp("2026-01-02 00:00", tz="Europe/Berlin")])
    forecast = _generation_frame("2026-01-01 00:00", 48)
    actual = _generation_frame("2026-01-01 00:00", 72)

    add_generation_features(target_times, market, forecast, actual, PUBLISHED_AT_07)

    assert "wind_onshore_forecast_mw" in FEATURE_AVAILABILITY
    assert "solar_forecast_mw" in FEATURE_AVAILABILITY
    assert f"wind_onshore_actual_lag{GENERATION_ACTUAL_LAG_HOURS}h_mw" in FEATURE_AVAILABILITY


def test_build_feature_frame_includes_generation_columns_when_provided():
    market = MARKETS["day_ahead"]
    target_times = pd.Series(pd.date_range("2026-01-02 00:00", periods=4, freq="1h", tz="Europe/Berlin"))
    target_series = pd.Series(np.arange(len(target_times), dtype=float), index=pd.DatetimeIndex(target_times))
    forecast = _generation_frame("2026-01-01 00:00", 48)
    cutoff = pd.Timestamp("2026-01-01 09:00", tz="Europe/Berlin")  # before PUBLISHED_AT_07 D-1... see below

    frame, dropped = build_feature_frame(
        market, cutoff, target_times, target_series,
        generation_forecast=forecast, generation_forecast_published_local=dt.time(0, 0),
    )
    assert "wind_onshore_forecast_mw" in frame.columns
    assert "wind_onshore_forecast_mw" not in dropped


def test_build_feature_frame_omits_generation_columns_when_not_provided():
    """No regression for existing callers: no generation args -> no new
    columns, not even empty ones."""
    market = MARKETS["day_ahead"]
    target_times = pd.Series(pd.date_range("2026-01-02 00:00", periods=4, freq="1h", tz="Europe/Berlin"))
    target_series = pd.Series(np.arange(len(target_times), dtype=float), index=pd.DatetimeIndex(target_times))
    cutoff = pd.Timestamp("2026-01-01 09:00", tz="Europe/Berlin")

    frame, _dropped = build_feature_frame(market, cutoff, target_times, target_series)
    assert not any("wind" in c or "solar" in c for c in frame.columns)
