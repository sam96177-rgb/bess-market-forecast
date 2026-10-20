"""Feature builder: assembles a feature frame for one market at one
cutoff, for a set of target delivery times, then enforces availability
(src.features.availability.filter_available_features) before returning —
no feature reaches the model without passing that check.

FCR/aFRR are built directly at their native 4-hour block granularity
(market.product_length_hours), never hourly-then-averaged: lag windows,
weather aggregation windows and calendar timestamps are all anchored to
the block start, so a "1 period ago" feature means "the previous 4h block"
for those markets, not "1 hour ago".

ENTSO-E's day-ahead LOAD forecast (src.ingest.entsoe.get_load_forecast) is
deliberately not wired in here: every one of this system's four markets
sets its forecast cutoff before 18:00 D-1 (day_ahead 11:00, fcr 07:00,
afrr 08:00), and ENTSO-E's own day-ahead forecasts don't publish until
18:00 D-1 (verified against the Transparency Platform's regulation text —
see src/features/availability.py). It would be dropped by the
availability filter on every single call, so wiring it in would only be
dead code that looks like a usable feature. The ingestion function itself
stays in src/ingest/entsoe.py in case a later, later-cutoff market (e.g.
intraday) can actually use it.

The day-ahead WIND/SOLAR generation forecast is the one exception
(PLAN_V6_REVISION.md V6-3): the user chose to use it anyway, under the
ASSUMED earlier publication time in config/markets.yaml's
`entsoe_generation_forecast_published_local` (see
src.config.load_entsoe_generation_forecast_published_local and
src.features.availability.daily_publication_rule) — see
add_generation_features below.
"""
from __future__ import annotations

import datetime as dt
import functools

import holidays as holidays_lib
import numpy as np
import pandas as pd

from src.config import MarketConfig
from src.features.availability import (
    calendar_day_lag_rule,
    daily_publication_rule,
    filter_available_features,
    lag_rule,
    lagged_actual_rule,
    register_feature_availability,
)

DE_HOLIDAYS = holidays_lib.Germany()

# Calendar-duration lags, expressed in DAYS (not hours): 1 day, 1 week, 4
# weeks ago — i.e. "same hour N days back". These are computed via
# pd.DateOffset (calendar-aware), not a fixed N*24h Timedelta — see
# add_lag_features for why a Timedelta silently breaks across DST.
LAG_CALENDAR_DAYS = [1, 7, 28]

# PLAN_V8_REVISION.md V8-4: LEAR's inputs are the whole price profile of these past days
# (Lago et al. 2021: 1, 2, 3 and 7 days back), as one column per hour (day-ahead) or per
# 4-hour block (FCR/aFRR). Only the `lear` model reads them (prefix LEAR_PREFIX).
PRICE_HISTORY_DAYS = [1, 2, 3, 7]
LEAR_PREFIX = "lear_p_"


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
    register_feature_availability(immediate_name, lag_rule(immediate_hours, market.results_published_local))

    for n_days in LAG_CALENDAR_DAYS:
        lag_name = f"lag_{n_days * 24}h"
        lookup_times = _shift_calendar_days(target_times, n_days)
        out[lag_name] = lookup_times.map(lookup)
        register_feature_availability(lag_name, calendar_day_lag_rule(n_days, market.results_published_local))

    return pd.DataFrame(out, index=target_times.index)


@functools.lru_cache(maxsize=None)
def _history_rule(n_days: int, results_published_local: dt.time):
    return calendar_day_lag_rule(n_days, results_published_local)


def add_price_history_features(
    target_times: pd.Series, target_series: pd.Series, market: MarketConfig
) -> pd.DataFrame:
    """One column per (past day, slot) holding that day's mean price in the slot, for every
    PRICE_HISTORY_DAYS lag: slots are clock hours (day-ahead) or the market's own blocks
    (FCR/aFRR). Every row of a delivery day gets the same profile. A slot with no price
    (a missing hour on the spring DST day, a data gap) stays NaN, never filled.

    A past day's prices are published the day before THAT day, so each column uses
    calendar_day_lag_rule, the same rule as the single-value lags above.
    """
    slot_hours = max(1, int(market.product_length_hours))
    n_slots = 24 // slot_hours
    tz = target_times.dt.tz
    delivery_days = pd.DatetimeIndex(target_times).tz_localize(None).normalize()

    profiles: dict[pd.Timestamp, np.ndarray] = {}
    for day in delivery_days.unique():
        parts = []
        for n_days in PRICE_HISTORY_DAYS:
            past = day - pd.DateOffset(days=n_days)
            day_start, day_end = past.tz_localize(tz), (past + pd.Timedelta(days=1)).tz_localize(tz)
            window = target_series[(target_series.index >= day_start) & (target_series.index < day_end)]
            slot = pd.Index(window.index.hour // slot_hours)
            parts.append(window.groupby(slot).mean().reindex(range(n_slots)).to_numpy())
        profiles[day] = np.concatenate(parts)

    names = [f"{LEAR_PREFIX}d{n_days}_s{slot:02d}" for n_days in PRICE_HISTORY_DAYS for slot in range(n_slots)]
    for n_days in PRICE_HISTORY_DAYS:
        rule = _history_rule(n_days, market.results_published_local)  # one object per lag day: evaluated once per frame
        for slot in range(n_slots):
            register_feature_availability(f"{LEAR_PREFIX}d{n_days}_s{slot:02d}", rule)
    values = np.vstack([profiles[day] for day in delivery_days])
    return pd.DataFrame(values, index=target_times.index, columns=names)


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


def hourly_to_quarter_hours(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Straight-line (time-weighted) fill of hourly data onto a 15-minute index
    (PLAN_V7_REVISION.md V7-1). A quarter-hour is left NaN, never filled across
    a gap, if the hour before or after it is missing; nothing is filled past the
    last known hour.
    """
    hourly_df = hourly_df.sort_index()
    quarters = pd.date_range(hourly_df.index[0], hourly_df.index[-1], freq="15min")
    filled = hourly_df.reindex(quarters).interpolate(method="time", limit_area="inside")
    # Floor in UTC: a local-time floor is ambiguous in the repeated 02:00 hour of the autumn clock change.
    hour_start = quarters.tz_convert("UTC").floor("h").tz_convert(quarters.tz)
    previous = hourly_df.reindex(hour_start).isna().to_numpy()
    following = hourly_df.reindex(hour_start + pd.Timedelta(hours=1)).isna().to_numpy()
    on_the_hour = (quarters.minute == 0)[:, None]
    return filled.mask(previous | (following & ~on_the_hour))


def join_block_mean(
    hourly_df: pd.DataFrame, target_times: pd.Series, block_length_hours: float
) -> pd.DataFrame:
    """Join data onto target_times, mean-aggregating over each
    [target_time, target_time + block_length_hours) window. When the data
    resolution equals block_length_hours (hourly data for an hourly block,
    15-minute data for the 0.25 h day-ahead block) this reduces to a lookup.

    Vectorised (PLAN_V5_REVISION.md V5-4): searchsorted finds every
    window's [start, end) row range in the sorted index in one call, then
    a cumulative-sum/cumulative-count trick gets each window's per-column
    mean without a per-target-time Python loop. NaN is skipped per
    column, matching pd.DataFrame.mean()'s default -- not per row.
    """
    hourly_df = hourly_df.sort_index()
    times = pd.DatetimeIndex(target_times)
    window = pd.Timedelta(hours=block_length_hours)

    start_pos = hourly_df.index.searchsorted(times, side="left")
    end_pos = hourly_df.index.searchsorted(times + window, side="left")

    values = hourly_df.to_numpy(dtype=float)
    n_cols = values.shape[1]
    is_valid = ~np.isnan(values)
    filled = np.where(is_valid, values, 0.0)

    zero_row = np.zeros((1, n_cols))
    sum_cum = np.vstack([zero_row, np.cumsum(filled, axis=0)])
    count_cum = np.vstack([zero_row, np.cumsum(is_valid, axis=0)])

    seg_sum = sum_cum[end_pos] - sum_cum[start_pos]
    seg_count = count_cum[end_pos] - count_cum[start_pos]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = seg_sum / seg_count
    means[seg_count == 0] = np.nan

    return pd.DataFrame(means, index=target_times.index, columns=hourly_df.columns)


# ENTSO-E PSR type column name -> this system's feature-name root.
_GENERATION_PSR_TO_ROOT = {"Wind Onshore": "wind_onshore", "Wind Offshore": "wind_offshore", "Solar": "solar"}

# PLAN_V6_REVISION.md V6-3's "past input": actual generation from this many
# hours before delivery ("two days before delivery").
GENERATION_ACTUAL_LAG_HOURS = 48


def add_generation_features(
    target_times: pd.Series,
    market: MarketConfig,
    generation_forecast: pd.DataFrame | None,
    generation_actual: pd.DataFrame | None,
    forecast_published_local: dt.time | None,
) -> pd.DataFrame:
    """PLAN_V6_REVISION.md V6-3: the day-ahead wind/solar generation
    forecast for the delivery period itself (future input, availability =
    forecast_published_local -- an ASSUMPTION, see module docstring), and
    actual generation from GENERATION_ACTUAL_LAG_HOURS before delivery
    (past input, availability = lagged_actual_rule -- a real, measured
    value, not an assumption). Both trained models (linear_lags,
    lgbm_point) get both; seasonal_naive ignores features entirely, so no
    per-model switch is needed here.
    """
    out: dict[str, np.ndarray] = {}

    if generation_forecast is not None and not generation_forecast.empty:
        if forecast_published_local is None:
            raise ValueError("generation_forecast was given but forecast_published_local is None")
        joined = join_block_mean(generation_forecast, target_times, market.product_length_hours)
        for psr, root in _GENERATION_PSR_TO_ROOT.items():
            if psr not in joined.columns:
                continue
            name = f"{root}_forecast_mw"
            out[name] = joined[psr].to_numpy()
            register_feature_availability(name, daily_publication_rule(forecast_published_local))

    if generation_actual is not None and not generation_actual.empty:
        lag_times = target_times - pd.Timedelta(hours=GENERATION_ACTUAL_LAG_HOURS)
        joined = join_block_mean(generation_actual, lag_times, market.product_length_hours)
        for psr, root in _GENERATION_PSR_TO_ROOT.items():
            if psr not in joined.columns:
                continue
            name = f"{root}_actual_lag{GENERATION_ACTUAL_LAG_HOURS}h_mw"
            out[name] = joined[psr].to_numpy()
            register_feature_availability(name, lagged_actual_rule(GENERATION_ACTUAL_LAG_HOURS))

    return pd.DataFrame(out, index=target_times.index)


def build_feature_frame(
    market: MarketConfig,
    cutoff_time: pd.Timestamp,
    target_times: pd.Series,
    target_series: pd.Series,
    weather: pd.DataFrame | None = None,
    generation_forecast: pd.DataFrame | None = None,
    generation_actual: pd.DataFrame | None = None,
    generation_forecast_published_local: dt.time | None = None,
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
    frame = pd.concat([frame, add_price_history_features(target_times, target_series, market)], axis=1)

    if weather is not None:
        if market.product_length_hours < 1:
            weather = hourly_to_quarter_hours(weather)
        joined = join_block_mean(weather, target_times, market.product_length_hours)
        for col in joined.columns:
            frame[col] = joined[col].values

    if generation_forecast is not None or generation_actual is not None:
        generation = add_generation_features(
            target_times, market, generation_forecast, generation_actual, generation_forecast_published_local,
        )
        for col in generation.columns:
            frame[col] = generation[col].values

    return filter_available_features(frame, target_time_col="target_time", cutoff_time=cutoff_time)
