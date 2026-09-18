"""Cutoff/target-window generation, driven entirely by config/markets.yaml.

No market-specific time constant (gate closure, cutoff, delivery start) may
be written here — every such value must come from a MarketConfig. This is
enforced by tests/test_backtest_no_hardcoded_timestamps.py.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from src.config import MarketConfig


def generate_cutoffs(market: MarketConfig, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """For each delivery day D in [start_date, end_date], produce one row per
    (horizon, target product window) pair: the cutoff time information is
    knowable by, and the target delivery window it forecasts.

    `delivery_day` is the calendar day the gate closure anchors to (D). The
    cutoff itself happens on `delivery_day + gate_closure_offset_days`, at
    `forecast_cutoff_local` — always before the market's gate closure.

    product_length_hours can be fractional (e.g. 0.25 for a 15-minute
    product — see day_ahead's config comment on the SDAC MTU change).

    European clock changes mean a calendar "day" isn't always 24h, and —
    verified empirically against each market's real published data, since
    this is NOT a hardcoded rule — the markets here don't handle it the
    same way (see markets.yaml's `dst_convention` comment for how each was
    confirmed):

    - "variable_period_count" (day_ahead): the day genuinely has fewer/
      more periods (23h -> 92 quarter-hours, 25h -> 100), matching real
      ENTSO-E data exactly. Generated with `pd.date_range` bounded to the
      calendar day, which naturally enumerates only the real timestamps
      that day contains.
    - "fixed_period_count" (fcr/afrr): always the nominal count (6 for a
      4h product); boundaries are this day's local midnight plus
      N*product_length_hours in absolute (pd.Timestamp/pd.Timedelta) time,
      matching regelleistung's real published blocks — including the
      last block's end spilling past next calendar day's midnight on a
      short day.

    Either way, boundaries are computed with pd.Timestamp/pd.Timedelta
    arithmetic (absolute time), never stdlib datetime/timedelta on a
    tz-attached value: the latter does wall-clock-naive arithmetic that
    silently duplicates or skips timestamps across a DST gap — caught by
    running a real 6-month backtest that happened to cross 2026-03-29.
    """
    raw_steps = 24 / market.product_length_hours
    n_steps = round(raw_steps)
    if not math.isclose(raw_steps, n_steps, rel_tol=1e-9):
        raise ValueError(
            f"{market.name}: product_length_hours={market.product_length_hours} "
            "must evenly divide 24"
        )
    product_freq = pd.Timedelta(hours=market.product_length_hours)

    rows = []
    delivery_day = start_date
    while delivery_day <= end_date:
        cutoff_date = delivery_day + dt.timedelta(days=market.gate_closure_offset_days)
        cutoff_time = dt.datetime.combine(cutoff_date, market.forecast_cutoff_local).replace(
            tzinfo=market.timezone
        )
        gate_closure_time = dt.datetime.combine(cutoff_date, market.gate_closure_local).replace(
            tzinfo=market.timezone
        )
        assert cutoff_time <= gate_closure_time, (
            f"{market.name}: forecast_cutoff_local must be before gate_closure_local"
        )

        for horizon_hours in market.horizons_hours:
            target_day = delivery_day + dt.timedelta(days=(horizon_hours // 24) - 1)
            # ambiguous=False: delivery_start_local is always midnight for
            # every market here, and European DST transitions happen at
            # 02:00-03:00, never at 00:00, so this is never actually
            # ambiguous — False is just a non-raising default in case a
            # future market's delivery_start_local ever lands differently.
            day_start = pd.Timestamp(
                dt.datetime.combine(target_day, market.delivery_start_local)
            ).tz_localize(market.timezone, ambiguous=False, nonexistent="shift_forward")

            if market.dst_convention == "variable_period_count":
                # pd.DateOffset (calendar-aware), NOT timedelta(days=1)
                # (absolute time) — the latter adds exactly 24 real hours,
                # landing an hour past next midnight on a short DST day.
                day_end = day_start + pd.DateOffset(days=1)
                boundaries = pd.date_range(day_start, day_end, freq=product_freq, inclusive="both")
            else:  # fixed_period_count
                boundaries = [day_start + step * product_freq for step in range(n_steps + 1)]

            for step in range(len(boundaries) - 1):
                rows.append(
                    {
                        "market": market.name,
                        "cutoff_time": cutoff_time,
                        "horizon_hours": horizon_hours,
                        "horizon_step": step + 1,
                        "target_window_start": boundaries[step],
                        "target_window_end": boundaries[step + 1],
                    }
                )
        delivery_day += dt.timedelta(days=1)

    return pd.DataFrame(rows)
