"""Feature builder: assembles a feature frame for one market at one
cutoff, for a set of target delivery times, then enforces availability
(src.features.availability.filter_available_features) before returning —
no feature reaches the model without passing that check.

FCR/aFRR are built directly at their native 4-hour block granularity
(market.product_length_hours), never hourly-then-averaged: lag windows,
weather aggregation windows and calendar timestamps are all anchored to
the block start, so a "1 period ago" feature means "the previous 4h block"
for those markets, not "1 hour ago".

ENTSO-E's day-ahead load/wind/solar forecasts (src.ingest.entsoe) are
deliberately not wired in here: every one of this system's four markets
sets its forecast cutoff before 18:00 D-1 (day_ahead 11:00, fcr 07:00,
afrr 08:00), and ENTSO-E's own day-ahead forecasts don't publish until
18:00 D-1 (verified against the Transparency Platform's regulation text —
see src/features/availability.py). They would be dropped by the
availability filter on every single call, so wiring them in would only be
dead code that looks like a usable feature. The ingestion functions
themselves stay in src/ingest/entsoe.py in case a later, later-cutoff
market (e.g. intraday) can actually use them.
"""
from __future__ import annotations

import holidays as holidays_lib
import pandas as pd

from src.config import MarketConfig
from src.features.availability import (
    calendar_day_lag_rule,
    filter_available_features,
    lag_rule,
    register_feature_availability,
)

DE_HOLIDAYS = holidays_lib.Germany()

# Calendar-duration lags, expressed in DAYS (not hours): 1 day, 1 week, 4
# weeks ago — i.e. "same hour N days back". These are computed via
# pd.DateOffset (calendar-aware), not a fixed N*24h Timedelta — see
# add_lag_features for why a Timedelta silently breaks across DST.
LAG_CALENDAR_DAYS = [1, 7, 28]


def add_calendar_features(target_times: pd.Series) -> pd.DataFrame:
    dt_idx = pd.DatetimeIndex(target_times)
    return pd.DataFrame(
        {
            "hour_of_day": dt_idx.hour,
            "day_of_week": dt_idx.dayofweek,
            "month": dt_idx.month,
            "is_weekend": (dt_idx.dayofweek >= 5).astype(int),
            "holidays": [1 if d in DE_HOLIDAYS else 0 for d in dt_idx.date],
            "is_winter": dt_idx.month.isin([12, 1, 2]).astype(int),
            "is_summer": dt_idx.month.isin([6, 7, 8]).astype(int),
        },
        index=target_times.index,
    )


def add_lag_features(
    target_times: pd.Series, target_series: pd.Series, market: MarketConfig
) -> pd.DataFrame:
    """target_series must be indexed by delivery timestamp (the settled/
    observed value for that period).

    Two different kinds of lag, computed two different ways:
    - The immediate-previous-period lag (one product length: 0.25h for
      day_ahead, 4h for fcr/afrr) is a genuinely short, contiguous-time
      gap — a fixed elapsed Timedelta is correct.
    - "Same hour N days/weeks ago" is a CALENDAR concept, not a fixed
      elapsed duration. Subtracting a flat N*24h Timedelta silently
      breaks it: after a DST transition removes/adds an hour, "168 real
      hours before this timestamp" lands on the wrong wall-clock hour,
      which then misses every lookup against the target series (itself
      indexed by wall-clock time) for days afterward — caught only by
      running a real 6-month backtest across 2026-03-29. pd.DateOffset
      shifts by calendar days instead, staying wall-clock-correct.
    """
    lookup = target_series.to_dict()
    out = {}

    immediate_hours = market.product_length_hours
    immediate_name = f"lag_{immediate_hours}h"
    out[immediate_name] = (target_times - pd.Timedelta(hours=immediate_hours)).map(lookup)
    register_feature_availability(immediate_name, lag_rule(immediate_hours))

    for n_days in LAG_CALENDAR_DAYS:
        lag_name = f"lag_{n_days * 24}h"
        lookup_times = _shift_calendar_days(target_times, n_days)
        out[lag_name] = lookup_times.map(lookup)
        register_feature_availability(lag_name, calendar_day_lag_rule(n_days))

    return pd.DataFrame(out, index=target_times.index)


def _shift_calendar_days(target_times: pd.Series, n_days: int) -> pd.Series:
    """Shift a tz-aware Series back by N calendar days. Series - DateOffset
    raises pytz.NonExistentTimeError outright if the result lands in a
    DST gap (e.g. 2024-03-31 02:00 in Europe/Berlin never happens) — there
    is no `nonexistent=` handling for that operator the way tz_localize
    has one. Fixed by dropping to naive (unambiguous calendar arithmetic),
    shifting, then re-attaching the tz with explicit gap handling.
    """
    naive = target_times.dt.tz_localize(None) - pd.DateOffset(days=n_days)
    return naive.dt.tz_localize(target_times.dt.tz, ambiguous=True, nonexistent="shift_forward")


def join_block_mean(
    hourly_df: pd.DataFrame, target_times: pd.Series, block_length_hours: int
) -> pd.DataFrame:
    """Join hourly-resolution data onto target_times, mean-aggregating
    over each [target_time, target_time + block_length_hours) window. For
    hourly markets (block_length_hours == 1) this reduces to a lookup."""
    hourly_df = hourly_df.sort_index()
    rows = []
    for t in target_times:
        window_end = t + pd.Timedelta(hours=block_length_hours) - pd.Timedelta(seconds=1)
        rows.append(hourly_df.loc[t:window_end].mean())
    return pd.DataFrame(rows, index=target_times.index)


def build_feature_frame(
    market: MarketConfig,
    cutoff_time: pd.Timestamp,
    target_times: pd.Series,
    target_series: pd.Series,
    weather: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature frame for `market` at `cutoff_time` for each
    target delivery time in `target_times`, then enforce availability.
    Returns (features_df, dropped_feature_names).
    """
    # NB: must not go through target_times.values here — for a tz-aware
    # Series, .values silently drops the tzinfo (converts to naive UTC),
    # which then breaks every availability comparison downstream.
    frame = pd.DataFrame({"target_time": target_times.reset_index(drop=True)})
    frame.index = target_times.index
    frame = pd.concat([frame, add_calendar_features(target_times)], axis=1)

    frame = pd.concat([frame, add_lag_features(target_times, target_series, market)], axis=1)

    if weather is not None:
        joined = join_block_mean(weather, target_times, market.product_length_hours)
        for col in joined.columns:
            frame[col] = joined[col].values

    return filter_available_features(frame, target_time_col="target_time", cutoff_time=cutoff_time)
