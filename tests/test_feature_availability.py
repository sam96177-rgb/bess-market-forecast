"""Tests for the availability-enforcement mechanism itself — the piece the
whole backtest's trustworthiness rests on. Includes a fixture that
reproduces the old notebook pipeline's known bug (a contemporaneous/actual
value used as a covariate) and asserts it gets caught.
"""
import datetime as dt
import logging

import pandas as pd
import pytest

from src.features.availability import (
    after_delivery_rule,
    always_available,
    calendar_day_lag_rule,
    daily_publication_rule,
    filter_available_features,
    lag_rule,
    lagged_actual_rule,
    register_feature_availability,
)

TZ = "Europe/Berlin"
PUBLISHED_AT_13 = dt.time(13, 0)  # a stand-in results_published_local for these unit tests


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
    register_feature_availability("lag_168h", lag_rule(168, PUBLISHED_AT_13))
    # lag's own delivery: 2024-05-27 00:00 -> published 2024-05-26 13:00, well before cutoff
    cutoff = _ts("2024-06-02 11:00")
    filtered, dropped = filter_available_features(df, "target_time", cutoff)
    assert dropped == []
    assert "lag_168h" in filtered.columns


def test_lagged_target_dropped_when_lag_lands_on_delivery_day(caplog):
    """PLAN_V5_REVISION.md V5-5: a lag whose own delivery still falls on
    delivery day D (not D-1 or earlier) is published on D-1 at
    results_published_local -- after the D-1 11:00 forecast cutoff, so it
    stays dropped. (This is the previous-period lag's real situation for
    every hour of the day except the first quarter-hour, where the 1h lag
    happens to reach back into D-1 -- see the next test.)"""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 08:00")],
        "lag_1h": [42.0],
    })
    register_feature_availability("lag_1h", lag_rule(1, PUBLISHED_AT_13))
    # lag's own delivery: 2024-06-03 07:00 (still day D) -> published 2024-06-02 13:00, AFTER cutoff
    cutoff = _ts("2024-06-02 11:00")
    filtered, dropped = filter_available_features(df, "target_time", cutoff)
    assert dropped == ["lag_1h"]
    assert "lag_1h" not in filtered.columns


def test_lag_24h_available_at_cutoff_under_new_publication_rule():
    """PLAN_V5_REVISION.md V5-5's actual fix: lag_24h's own delivery lands
    on D-1, so it's published D-2 at results_published_local -- always
    well before a D-1 cutoff. This is the case the old rule (available at
    the lag's own delivery time, i.e. D-1's actual hour) got wrong."""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 08:00")],  # delivery day D
        "lag_24h": [42.0],
    })
    register_feature_availability("lag_24h", calendar_day_lag_rule(1, PUBLISHED_AT_13))
    cutoff = _ts("2024-06-02 11:00")  # day-ahead's real forecast_cutoff_local
    filtered, dropped = filter_available_features(df, "target_time", cutoff)
    assert dropped == []
    assert "lag_24h" in filtered.columns


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


# ---- PLAN_V6_REVISION.md V6-3: wind/solar generation features -------------

def test_generation_forecast_kept_under_the_assumed_07_00_time_but_dropped_under_18_00():
    """The same feature, scored against the ASSUMED (07:00) vs. the
    VERIFIED (18:00) ENTSO-E publication time: the assumption is what makes
    day_ahead's 11:00 cutoff able to use it at all."""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 00:00")],  # delivery day D
        "wind_onshore_forecast_mw": [123.0],
    })
    day_ahead_cutoff = _ts("2024-06-02 11:00")  # day_ahead's real forecast_cutoff_local

    register_feature_availability("wind_onshore_forecast_mw", daily_publication_rule(dt.time(7, 0)))
    _, dropped = filter_available_features(df, "target_time", day_ahead_cutoff)
    assert dropped == []

    register_feature_availability("wind_onshore_forecast_mw", daily_publication_rule(dt.time(18, 0)))
    _, dropped = filter_available_features(df, "target_time", day_ahead_cutoff)
    assert dropped == ["wind_onshore_forecast_mw"]


def test_lagged_actual_generation_available_once_its_own_lag_slot_has_elapsed():
    """A 48h-lagged actual generation value is available once the LAG's own
    delivery period has elapsed -- not target_time's (future) delivery."""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 10:00")],
        "wind_onshore_actual_lag48h_mw": [55.0],
    })
    register_feature_availability("wind_onshore_actual_lag48h_mw", lagged_actual_rule(48, delivery_length_hours=1))
    # lag's own delivery: 2024-06-01 10:00, elapses at 2024-06-01 11:00 -- well before this cutoff.
    cutoff = _ts("2024-06-02 11:00")
    _, dropped = filter_available_features(df, "target_time", cutoff)
    assert dropped == []


def test_lagged_actual_generation_rejected_if_the_lag_itself_is_still_in_the_future():
    """A naive after_delivery_rule on target_time directly would wrongly
    accept this (target_time + 1h is also after the cutoff's own target day
    starts); lagged_actual_rule must anchor to the LAG's own slot instead."""
    df = pd.DataFrame({
        "target_time": [_ts("2024-06-03 10:00")],
        "wind_onshore_actual_lag48h_mw": [55.0],
    })
    register_feature_availability("wind_onshore_actual_lag48h_mw", lagged_actual_rule(48, delivery_length_hours=1))
    # A much smaller lag depth would put the "lag slot" after the cutoff.
    register_feature_availability("wind_onshore_actual_lag1h_mw", lagged_actual_rule(1, delivery_length_hours=1))
    df["wind_onshore_actual_lag1h_mw"] = [55.0]
    cutoff = _ts("2024-06-02 11:00")
    _, dropped = filter_available_features(df, "target_time", cutoff)
    assert "wind_onshore_actual_lag1h_mw" in dropped
    assert "wind_onshore_actual_lag48h_mw" not in dropped


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
