"""Tests for the availability-enforcement mechanism itself — the piece the
whole backtest's trustworthiness rests on. Includes a fixture that
reproduces the old notebook pipeline's known bug (a contemporaneous/actual
value used as a covariate) and asserts it gets caught.
"""
import logging

import pandas as pd
import pytest

from src.features.availability import (
    after_delivery_rule,
    always_available,
    daily_publication_rule,
    filter_available_features,
    lag_rule,
    register_feature_availability,
)

TZ = "Europe/Berlin"


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


def test_calendar_feature_always_survives():
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 00:00"), _ts("2024-06-03 01:00")],
        "hour_of_day": [0, 1],
    })
    register_feature_availability("hour_of_day", always_available)
    filtered, dropped = filter_available_features(df, "target_time", _ts("2024-06-02 11:00"))
    assert dropped == []
    assert "hour_of_day" in filtered.columns


def test_lagged_target_available_when_lag_precedes_cutoff():
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 00:00")],
        "lag_168h": [42.0],
    })
    register_feature_availability("lag_168h", lag_rule(168))
    cutoff = _ts("2024-06-02 11:00")  # target - 168h = 2024-05-27 00:00, well before cutoff
    filtered, dropped = filter_available_features(df, "target_time", cutoff)
    assert dropped == []
    assert "lag_168h" in filtered.columns


def test_lagged_target_dropped_when_lag_is_too_short_for_day_ahead_cutoff():
    """This is the real day-ahead constraint discovered while building
    this feature: lag_1h (previous hour's actual price) is NEVER usable
    for a day-ahead forecast, because target_time - 1h still falls after
    the D-1 11:00 cutoff for every hour of the delivery day."""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 00:00")],
        "lag_1h": [42.0],
    })
    register_feature_availability("lag_1h", lag_rule(1))
    cutoff = _ts("2024-06-02 11:00")  # target - 1h = 2024-06-02 23:00, AFTER cutoff
    filtered, dropped = filter_available_features(df, "target_time", cutoff)
    assert dropped == ["lag_1h"]
    assert "lag_1h" not in filtered.columns


def test_leaking_contemporaneous_actual_is_caught():
    """Reproduces the old notebook pipeline's known leakage bug directly:
    a contemporaneous *actual* value (only knowable after its own delivery
    period ends) used naively as if it were a predictor for that same
    period. This is exactly the bug the availability mechanism exists to
    catch structurally, not by convention.
    """
    target_time = _ts("2024-06-03 00:00")
    df = pd.DataFrame({
        "target_time": [target_time],
        "hour_of_day": [0],
        "actual_temperature_leak": [12.3],  # contemporaneous actual, wrongly used as a feature
    })
    register_feature_availability("hour_of_day", always_available)
    register_feature_availability("actual_temperature_leak", after_delivery_rule(delivery_length_hours=1))

    cutoff = _ts("2024-06-02 11:00")  # normal day-ahead cutoff, well before target_time even starts
    filtered, dropped = filter_available_features(df, "target_time", cutoff)

    assert "actual_temperature_leak" in dropped
    assert "actual_temperature_leak" not in filtered.columns
    assert "hour_of_day" in filtered.columns  # legitimate feature must survive


def test_leak_is_logged_as_a_warning(caplog):
    target_time = _ts("2024-06-03 00:00")
    df = pd.DataFrame({
        "target_time": [target_time],
        "actual_leak": [1.0],
    })
    register_feature_availability("actual_leak", after_delivery_rule(1))
    with caplog.at_level(logging.WARNING):
        filter_available_features(df, "target_time", _ts("2024-06-02 11:00"))
    assert any("actual_leak" in record.message for record in caplog.records)


def test_unregistered_feature_is_dropped_fail_closed():
    """A feature with no availability rule at all must be treated as
    unavailable, not silently allowed through."""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 00:00")],
        "mystery_column": [1.0],
    })
    filtered, dropped = filter_available_features(df, "target_time", _ts("2024-06-02 11:00"))
    assert dropped == ["mystery_column"]


def test_entsoe_forecast_available_before_gate_closure_but_not_before_publication():
    """ENTSO-E day-ahead load/wind/solar forecasts publish D-1 18:00 —
    after the day-ahead auction's own D-1 11:00 forecast_cutoff_local from
    Milestone 1. This asserts that mismatch is caught: at the market's own
    forecast cutoff, ENTSO-E's forecast for the *next* delivery day isn't
    published yet."""
    rule = daily_publication_rule(pd.Timestamp("18:00").time())
    target_time = _ts("2024-06-03 00:00")  # delivery day D
    published_at = rule(target_time)
    assert published_at == _ts("2024-06-02 18:00")  # D-1 18:00

    day_ahead_cutoff = _ts("2024-06-02 11:00")  # Milestone 1's forecast_cutoff_local
    assert published_at > day_ahead_cutoff, (
        "ENTSO-E's own forecast publishes AFTER the day-ahead market's forecast "
        "cutoff — a real constraint this system must respect, not an error."
    )
