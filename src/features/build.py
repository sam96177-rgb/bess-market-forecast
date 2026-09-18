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
    filter_available_features,
    lag_rule,
    register_feature_availability,
)

DE_HOLIDAYS = holidays_lib.Germany()

# Calendar-duration lags (hours), independent of market granularity: the
# immediate previous period, 1 day, 1 week, 4 weeks ago.
LAG_DURATIONS_HOURS = [24, 168, 672]


def lag_hours_for_market(market: MarketConfig) -> list[int]:
    """The immediate-previous-period lag is one product length for this
    market (1h for day_ahead, 4h for fcr/afrr); the calendar-duration lags
    (1d/1w/4w) are the same across markets."""
    hours = [market.product_length_hours] + LAG_DURATIONS_HOURS
    return sorted(set(hours))


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
    target_times: pd.Series, target_series: pd.Series, lag_hours_list: list[int]
) -> pd.DataFrame:
    """target_series must be indexed by delivery timestamp (the settled/
    observed value for that period)."""
    lookup = target_series.to_dict()
    out = {}
    for lag in lag_hours_list:
        lag_name = f"lag_{lag}h"
        lookup_times = target_times - pd.Timedelta(hours=lag)
        out[lag_name] = lookup_times.map(lookup)
        register_feature_availability(lag_name, lag_rule(lag))
    return pd.DataFrame(out, index=target_times.index)


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

    lag_hours = lag_hours_for_market(market)
    frame = pd.concat([frame, add_lag_features(target_times, target_series, lag_hours)], axis=1)

    if weather is not None:
        joined = join_block_mean(weather, target_times, market.product_length_hours)
        for col in joined.columns:
            frame[col] = joined[col].values

    return filter_available_features(frame, target_time_col="target_time", cutoff_time=cutoff_time)
