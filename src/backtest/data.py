"""Per-market data loading: dispatches to the right ingestion function and
returns a target Series indexed by delivery timestamp, uniform across
markets, for the backtest engine to consume.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src.ingest.entsoe import get_day_ahead_prices, get_wind_and_solar_actual, get_wind_and_solar_forecast
from src.ingest.regelleistung import get_afrr_capacity_prices, get_fcr_capacity_prices
from src.ingest.weather import get_capacity_weighted_weather

logger = logging.getLogger(__name__)


def _warn_failures(source: str, failures) -> None:
    if failures:
        logger.warning("%s ingestion had failed months: %s", source, failures)


def load_target_series(market_name: str, start_date: dt.date, end_date: dt.date) -> pd.Series:
    if market_name == "day_ahead":
        df, failures = get_day_ahead_prices(start_date, end_date)
        _warn_failures("day_ahead", failures)
        return df["da_price_eur_mwh"]

    if market_name == "fcr":
        df, failures = get_fcr_capacity_prices(start_date, end_date)
        _warn_failures("fcr", failures)
        return df.set_index("delivery_start")["fcr_price_eur_mw"]

    if market_name in ("afrr_pos", "afrr_neg"):
        direction = market_name.split("_")[1]
        df, failures = get_afrr_capacity_prices(start_date, end_date)
        _warn_failures("afrr", failures)
        df = df[df["direction"] == direction]
        return df.set_index("delivery_start")["afrr_capacity_eur_mw"]

    raise ValueError(f"Unknown market: {market_name}")


def load_weather(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    df, failures = get_capacity_weighted_weather(start_date, end_date)
    _warn_failures("weather", failures)
    return df.set_index("datetime")


def load_wind_and_solar_forecast(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """PLAN_V6_REVISION.md V6-3's future input, indexed by delivery quarter-hour
    (load_weather stays hourly) -- columns are ENTSO-E's own PSR type names ("Wind
    Onshore", "Wind Offshore", "Solar"), consumed by
    src.features.build.add_generation_features. Native 15-minute series
    (PLAN_V7_REVISION.md V7-1): day-ahead gets one value per quarter-hour,
    4-hour markets still get the block mean."""
    df, failures = get_wind_and_solar_forecast(start_date, end_date, resample=False)
    _warn_failures("wind_and_solar_forecast", failures)
    return df


def load_wind_and_solar_actual(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """PLAN_V6_REVISION.md V6-3's past input -- same shape as
    load_wind_and_solar_forecast, but realised generation. Never use a row
    here as if it were known at its own delivery time: that is exactly what
    src.features.availability.lagged_actual_rule exists to prevent."""
    df, failures = get_wind_and_solar_actual(start_date, end_date, resample=False)
    _warn_failures("wind_and_solar_actual", failures)
    return df
