"""Cutoff/target-window generation, driven entirely by config/markets.yaml.

No market-specific time constant (gate closure, cutoff, delivery start) may
be written here — every such value must come from a MarketConfig. This is
enforced by tests/test_backtest_no_hardcoded_timestamps.py.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.config import MarketConfig


def generate_cutoffs(market: MarketConfig, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """For each delivery day D in [start_date, end_date], produce one row per
    (horizon, target product window) pair: the cutoff time information is
    knowable by, and the target delivery window it forecasts.

    `delivery_day` is the calendar day the gate closure anchors to (D). The
    cutoff itself happens on `delivery_day + gate_closure_offset_days`, at
    `forecast_cutoff_local` — always before the market's gate closure.
    """
    n_steps = 24 // market.product_length_hours
    if n_steps * market.product_length_hours != 24:
        raise ValueError(
            f"{market.name}: product_length_hours={market.product_length_hours} "
            "must evenly divide 24"
        )

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
            for step in range(n_steps):
                hour_offset = step * market.product_length_hours
                window_start = dt.datetime.combine(target_day, market.delivery_start_local).replace(
                    tzinfo=market.timezone
                ) + dt.timedelta(hours=hour_offset)
                window_end = window_start + dt.timedelta(hours=market.product_length_hours)
                rows.append(
                    {
                        "market": market.name,
                        "cutoff_time": cutoff_time,
                        "horizon_hours": horizon_hours,
                        "horizon_step": step + 1,
                        "target_window_start": window_start,
                        "target_window_end": window_end,
                    }
                )
        delivery_day += dt.timedelta(days=1)

    return pd.DataFrame(rows)
