"""Availability metadata for features. For a target delivery time,
`available_at` returns the earliest wall-clock moment that feature's value
is knowable in reality. `filter_available_features` is the enforcement
mechanism that makes every backtest cutoff trustworthy: given a cutoff, it
drops — loudly, with a logged warning naming each one — every feature not
yet available at that cutoff, rather than silently including it.

This is the single mechanism meant to catch the old notebook pipeline's
leakage bug (a contemporaneous/actual value used as if it were knowable
ahead of time). See tests/test_feature_availability.py for a fixture that
reproduces exactly that bug and asserts it gets caught.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

AvailabilityRule = Callable[[pd.Timestamp], pd.Timestamp]

_ALWAYS = pd.Timestamp("1970-01-01", tz="UTC")


def always_available(target_time: pd.Timestamp) -> pd.Timestamp:
    """Calendar-type features (hour, weekday, holidays, ...): knowable
    regardless of cutoff."""
    return _ALWAYS


def _day_before_publication(lag_delivery_time: pd.Timestamp, results_published_local: dt.time) -> pd.Timestamp:
    """PLAN_V5_REVISION.md V5-5: a lag feature's own value is an auction
    price, published the day before ITS delivery day at the market's
    results_published_local — not "known instantly at its own delivery
    time" (the bug the old rule had, which made lag_24h look unavailable
    at every reasonable cutoff)."""
    publish_date = lag_delivery_time.date() - dt.timedelta(days=1)
    naive = dt.datetime.combine(publish_date, results_published_local)
    return pd.Timestamp(naive).tz_localize(lag_delivery_time.tzinfo, ambiguous=True, nonexistent="shift_forward")


def lag_rule(lag_hours: float, results_published_local: dt.time) -> AvailabilityRule:
    """Lagged target values: the lag itself is `lag_hours` before
    target_time (a fixed elapsed Timedelta — correct for a short,
    contiguous-time lag like "the immediately preceding delivery block",
    NOT a calendar-duration lag like "same hour last week"; use
    calendar_day_lag_rule for those). Available the day before THAT lag's
    own delivery day, at results_published_local — see
    _day_before_publication."""
    def rule(target_time: pd.Timestamp) -> pd.Timestamp:
        lag_delivery_time = target_time - pd.Timedelta(hours=lag_hours)
        return _day_before_publication(lag_delivery_time, results_published_local)
    return rule


def calendar_day_lag_rule(n_days: int, results_published_local: dt.time) -> AvailabilityRule:
    """Calendar-duration lags ("same hour N days/weeks ago"): the lag's own
    delivery time is target_time minus N calendar days, via pd.DateOffset
    (wall-clock-aware), not a fixed N*24h Timedelta. The two differ
    whenever the window spans a DST transition — a Timedelta shifts the
    wall-clock hour by one for several days after every spring/autumn
    clock change, which then mismatches the target series' own wall-
    clock-indexed lookup keys and makes the feature availability rule lie
    about when the value is really knowable. Must match
    src.features.build's add_lag_features lookup exactly, or the two
    silently disagree. Available the day before that lag's own delivery
    day, at results_published_local — see _day_before_publication.
    """
    def rule(target_time: pd.Timestamp) -> pd.Timestamp:
        # Timestamp - DateOffset raises pytz.NonExistentTimeError outright
        # if the result lands in a DST gap (no `nonexistent=` handling for
        # that operator) — drop to naive, shift, then re-attach the tz
        # with explicit gap handling, matching src.features.build's
        # _shift_calendar_days exactly (they must agree).
        tz = target_time.tzinfo
        naive = target_time.tz_localize(None) - pd.DateOffset(days=n_days)
        lag_delivery_time = naive.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        return _day_before_publication(lag_delivery_time, results_published_local)
    return rule


def daily_publication_rule(
    publish_time: dt.time, day_offset: int = -1, timezone: str = "Europe/Berlin"
) -> AvailabilityRule:
    """Features published once per day for the following delivery day(s).

    Verified against the ENTSO-E Transparency Platform's own regulation
    text (retrieved 2026-09-17) for both [6.1.B] Day-ahead Total Load
    Forecast and [14.1.D] Generation Forecasts for Wind and Solar: "The
    information shall be published no later than 18.00 Brussels time, one
    day before actual delivery takes place." Brussels time is CET/CEST,
    same wall clock as Europe/Berlin, so default publish_time=18:00,
    day_offset=-1 applies to both.
    """
    def rule(target_time: pd.Timestamp) -> pd.Timestamp:
        publish_date = target_time.date() + dt.timedelta(days=day_offset)
        return pd.Timestamp.combine(publish_date, publish_time).tz_localize(timezone)
    return rule


def after_delivery_rule(delivery_length_hours: float = 1) -> AvailabilityRule:
    """Observed/actual series: available only once the delivery period
    they describe has fully elapsed — never at or before its start."""
    def rule(target_time: pd.Timestamp) -> pd.Timestamp:
        return target_time + pd.Timedelta(hours=delivery_length_hours)
    return rule


def lagged_actual_rule(lag_hours: float, delivery_length_hours: float = 1) -> AvailabilityRule:
    """A column holding an OBSERVED/actual value from `lag_hours` before
    target_time (PLAN_V6_REVISION.md V6-3's "past input": actual wind/solar
    generation two days — lag_hours=48 — before delivery). Available once
    that lag's own delivery period has elapsed, per after_delivery_rule —
    anchored to the lag's own delivery time via lag_hours first, the same
    way lag_rule anchors to the lag's own delivery time before checking
    _day_before_publication; a bare after_delivery_rule on target_time
    itself would check the wrong (future) slot's availability."""
    def rule(target_time: pd.Timestamp) -> pd.Timestamp:
        lag_delivery_time = target_time - pd.Timedelta(hours=lag_hours)
        return lag_delivery_time + pd.Timedelta(hours=delivery_length_hours)
    return rule


# name -> rule, for every feature this system currently produces. An
# unregistered feature is NOT assumed safe — filter_available_features
# treats "no rule" as "unavailable" (fail closed).
FEATURE_AVAILABILITY: dict[str, AvailabilityRule] = {
    "hour_of_day": always_available,
    "day_of_week": always_available,
    "month": always_available,
    "is_weekend": always_available,
    "holidays": always_available,
    "is_winter": always_available,
    "is_summer": always_available,
    # ENTSO-E's day-ahead load/wind/solar forecasts are NOT registered here
    # on purpose: verified publication is D-1 18:00, after every market's
    # forecast cutoff in this system (day_ahead 11:00, fcr 07:00, afrr
    # 08:00), so they would be dropped on every call. See
    # src/features/build.py's module docstring for the full reasoning —
    # the ingestion functions stay in src/ingest/entsoe.py regardless, in
    # case a later, later-cutoff market can use them.
    #
    # Open-Meteo's Historical Forecast API doesn't expose the exact issue
    # time of each archived forecast, so this is an ASSUMPTION, not a
    # verified fact like the ENTSO-E deadlines above — but it's grounded in
    # real evidence, not copied from ENTSO-E's unrelated cadence: Open-
    # Meteo's own docs (retrieved 2026-09-17) state global models update
    # every 6h, and their GFS 00z run — first initialised at 00:00 GMT —
    # is fully ingested by ~05:15 GMT (+10 min latency), i.e. roughly
    # 06:00-07:30 CET/CEST. DWD ICON (Open-Meteo's likely "best_match" pick
    # for Germany) runs on a similar 4x-daily cadence. A next-day forecast
    # is therefore realistically available from early morning D-1, not
    # evening D-1. Rounded up (later, i.e. more conservative) to D-1 08:00
    # for a safety margin: this clears day_ahead (11:00) and aFRR (08:00,
    # exactly at the boundary) but NOT fcr (07:00) — which is a plausible
    # real constraint (FCR bidders commit before the morning weather run
    # lands), not an artifact of an arbitrary assumption.
    "temperature_2m_c": daily_publication_rule(dt.time(8, 0)),
    "precipitation_mm": daily_publication_rule(dt.time(8, 0)),
    "rain_mm": daily_publication_rule(dt.time(8, 0)),
    "snowfall_cm": daily_publication_rule(dt.time(8, 0)),
    "cloud_cover_pct": daily_publication_rule(dt.time(8, 0)),
    "wind_speed_10m_kmh": daily_publication_rule(dt.time(8, 0)),
    "wind_speed_100m_kmh": daily_publication_rule(dt.time(8, 0)),
    "shortwave_radiation_w_m2": daily_publication_rule(dt.time(8, 0)),
}


def register_feature_availability(name: str, rule: AvailabilityRule) -> None:
    """For features not fixed in the static registry above — currently
    just the per-market target lags, whose lag hours depend on the
    market's product_length_hours (see src/features/build.py)."""
    FEATURE_AVAILABILITY[name] = rule


def filter_available_features(
    df: pd.DataFrame,
    target_time_col: str,
    cutoff_time: pd.Timestamp,
    registry: dict[str, AvailabilityRule] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop every feature column not yet available at cutoff_time for its
    row's target_time. A feature is dropped if it is unavailable for ANY
    row being built (fail closed, not row-by-row NaN'ing) — if a feature
    can't be trusted for one target in the batch, it isn't trusted for the
    batch. Returns (filtered_df, dropped_feature_names); every drop is
    also logged as a warning naming the feature and the violating row.
    """
    registry = registry if registry is not None else FEATURE_AVAILABILITY
    feature_cols = [c for c in df.columns if c != target_time_col]
    target_times = df[target_time_col]

    dropped: list[str] = []
    availability_by_rule: dict[AvailabilityRule, pd.Series] = {}  # columns sharing one rule object evaluate it once
    for col in feature_cols:
        rule = registry.get(col)
        if rule is None:
            logger.warning(
                "Dropping feature '%s': no availability rule registered for it. "
                "Register one in FEATURE_AVAILABILITY before using this feature.",
                col,
            )
            dropped.append(col)
            continue

        if rule not in availability_by_rule:
            availability_by_rule[rule] = target_times.map(rule)
        available_ats = availability_by_rule[rule]
        violations = available_ats > cutoff_time
        if violations.any():
            first_bad = violations.idxmax()
            logger.warning(
                "Dropping feature '%s': not available until %s, after cutoff %s "
                "(first violation at target_time=%s). This would leak future "
                "information into the forecast.",
                col, available_ats.loc[first_bad], cutoff_time, target_times.loc[first_bad],
            )
            dropped.append(col)

    kept = [c for c in df.columns if c not in dropped]
    return df[kept], dropped
