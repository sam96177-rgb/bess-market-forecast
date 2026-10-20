"""Regression test for a real bug: 'same hour N days/weeks ago' lag
features were computed with a fixed N*24h Timedelta, which silently
misaligns by an hour for dates on the other side of a DST transition
from their reference point — since pd.Timestamp arithmetic with a
Timedelta operates in absolute elapsed time, not wall-clock/calendar
time. For a market with gap-free data at every timestamp (like
day_ahead's 15-min grid) this produces a SILENTLY WRONG lag value, not
even a visible NaN. For FCR specifically (irregular block boundaries on
the transition day itself), it produced NaN for every seasonal_naive
prediction across the whole week following 2026-03-29 in a real 6-month
backtest. pd.DateOffset (calendar-aware) must be used instead.
"""
import datetime as dt

import pandas as pd

from src.backtest.data import load_target_series
from src.config import load_markets_config
from src.features.build import add_lag_features

MARKETS = load_markets_config()


def test_timedelta_and_dateoffset_genuinely_disagree_across_the_transition():
    """Sanity-checks the premise: for a target the day after the 2026
    spring-forward transition, 'exactly 168 real hours ago' and '7
    calendar days ago' land on different UTC instants entirely (not just
    different formatting) — this is the disagreement the fix resolves."""
    target = pd.Timestamp("2026-03-30 00:00", tz="Europe/Berlin")
    via_timedelta = target - pd.Timedelta(hours=168)
    via_dateoffset = target - pd.DateOffset(days=7)
    assert via_timedelta.tz_convert("UTC") != via_dateoffset.tz_convert("UTC")
    assert via_dateoffset.date() == dt.date(2026, 3, 23)  # correct: exactly 7 calendar days back
    assert via_timedelta.date() == dt.date(2026, 3, 22)  # the bug: lands a whole day earlier


def test_day_ahead_week_ago_lag_is_correct_not_just_non_null_across_dst():
    """day_ahead's 15-min grid has a value at every timestamp, so the old
    bug wouldn't even show up as NaN here — it would silently return the
    WRONG hour's price. Assert the lag value matches the real price at
    the calendar-correct instant, not the Timedelta-shifted one."""
    market = MARKETS["day_ahead"]
    series = load_target_series("day_ahead", dt.date(2026, 3, 15), dt.date(2026, 4, 6))

    target_times = pd.Series([pd.Timestamp("2026-03-30 00:00", tz="Europe/Berlin")])
    lags = add_lag_features(target_times, series, market)

    correct_value = series[pd.Timestamp("2026-03-23 00:00", tz="Europe/Berlin")]
    wrong_value_if_bug_present = series.get(pd.Timestamp("2026-03-22 23:00", tz="Europe/Berlin"))

    assert lags["lag_168h"].iloc[0] == correct_value
    if wrong_value_if_bug_present is not None:
        assert lags["lag_168h"].iloc[0] != wrong_value_if_bug_present


def test_fcr_week_ago_lag_recovers_from_the_dst_week_wipeout():
    """The real regression this was caught by: every FCR seasonal_naive
    prediction (which is purely lag_168h) came back NaN for the entire
    week following 2026-03-29 before this fix."""
    market = MARKETS["fcr"]
    series = load_target_series("fcr", dt.date(2026, 3, 1), dt.date(2026, 4, 10))

    target_times = pd.Series([
        pd.Timestamp("2026-03-30 00:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-03-31 04:00", tz="Europe/Berlin"),
        pd.Timestamp("2026-04-02 08:00", tz="Europe/Berlin"),  # FCR blocks are 00/04/08/12/16/20
    ])
    lags = add_lag_features(target_times, series, market)
    assert lags["lag_168h"].notna().all()


def test_lag_names_and_availability_are_registered_consistently():
    from src.features.availability import FEATURE_AVAILABILITY

    market = MARKETS["day_ahead"]
    series = load_target_series("day_ahead", dt.date(2026, 1, 1), dt.date(2026, 2, 1))
    target_times = pd.Series([pd.Timestamp("2026-01-15 12:00", tz="Europe/Berlin")])
    add_lag_features(target_times, series, market)

    for name in ["lag_0.25h", "lag_24h", "lag_168h", "lag_672h"]:
        assert name in FEATURE_AVAILABILITY, f"{name} should have been registered by add_lag_features"


def test_lag_24h_clears_every_markets_real_cutoff():
    """PLAN_V5_REVISION.md V5-5's actual claim: with the new publication-
    based rule and each market's real config/markets.yaml
    results_published_local, lag_24h is available at that market's own
    forecast_cutoff_local -- for every one of the 4 markets, not just
    day_ahead."""
    from src.features.availability import FEATURE_AVAILABILITY

    delivery_day = dt.date(2026, 6, 3)  # ordinary day, no DST nearby
    for market_name, market in MARKETS.items():
        series = load_target_series(market_name, delivery_day - dt.timedelta(days=60), delivery_day)
        target_time = pd.Timestamp.combine(delivery_day, market.delivery_start_local).tz_localize(market.timezone)
        target_times = pd.Series([target_time])
        add_lag_features(target_times, series, market)

        cutoff_date = delivery_day + dt.timedelta(days=market.gate_closure_offset_days)
        cutoff = pd.Timestamp.combine(cutoff_date, market.forecast_cutoff_local).tz_localize(market.timezone)

        available_at = FEATURE_AVAILABILITY["lag_24h"](target_time)
        assert available_at <= cutoff, (
            f"{market_name}: lag_24h available at {available_at}, after its own cutoff {cutoff}"
        )
