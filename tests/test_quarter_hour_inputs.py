"""PLAN_V7_REVISION.md V7-1: day-ahead models get a value on every quarter-hour
from weather (straight-line fill) and wind/solar (native 15-minute data), and
4-hour markets still get the same block means as before.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import load_markets_config
from src.features.build import build_feature_frame, hourly_to_quarter_hours, join_block_mean

MARKETS = load_markets_config()
TZ = "Europe/Berlin"


def _hourly(values: list[float], start: str = "2026-01-01 00:00") -> pd.DataFrame:
    return pd.DataFrame({"temperature_2m_c": values}, index=pd.date_range(start, periods=len(values), freq="1h", tz=TZ))


def test_quarter_hours_are_a_straight_line_between_the_hours():
    out = hourly_to_quarter_hours(_hourly([0.0, 4.0, 8.0]))
    assert out["temperature_2m_c"].tolist() == pytest.approx([0, 1, 2, 3, 4, 5, 6, 7, 8])


def test_nothing_is_filled_past_the_last_hour_or_across_a_missing_hour():
    out = hourly_to_quarter_hours(_hourly([0.0, 4.0, np.nan, 8.0, 12.0]))["temperature_2m_c"]
    assert len(out) == 17  # ends at the last known hour, no :15/:30/:45 after it
    # hour 2 is missing: its own quarter and the quarters either side of it stay blank, not bridged
    assert out.iloc[4] == 4.0 and out.iloc[5:8].isna().all() and out.iloc[8:12].isna().all()
    assert out.iloc[12] == 8.0 and out.iloc[13] == pytest.approx(9.0)


def test_day_ahead_feature_frame_has_weather_and_generation_on_every_quarter_hour():
    market = MARKETS["day_ahead"]
    target_times = pd.Series(pd.date_range("2026-01-02 00:00", periods=96, freq="15min", tz=TZ))
    target_series = pd.Series(np.arange(96, dtype=float), index=pd.DatetimeIndex(target_times))
    weather = _hourly(list(np.arange(72, dtype=float)))
    quarters = pd.date_range("2026-01-01 00:00", periods=72 * 4 - 3, freq="15min", tz=TZ)
    generation = pd.DataFrame({"Wind Onshore": np.arange(len(quarters), dtype=float), "Solar": 5.0}, index=quarters)

    frame, _ = build_feature_frame(
        market, pd.Timestamp("2026-01-01 12:00", tz=TZ), target_times, target_series, weather=weather,
        generation_forecast=generation, generation_forecast_published_local=dt.time(0, 0),
    )
    for column in ("temperature_2m_c", "wind_onshore_forecast_mw", "solar_forecast_mw"):
        assert frame[column].notna().all(), column
    # :15 differs from :00 (before the fix it was blank and then 0.0 in linear_lags)
    assert frame["temperature_2m_c"].iloc[1] != frame["temperature_2m_c"].iloc[0]


def test_four_hour_block_means_do_not_change_with_15_minute_generation_data():
    hourly_index = pd.date_range("2026-01-01 00:00", periods=24, freq="1h", tz=TZ)
    quarter_index = pd.date_range("2026-01-01 00:00", periods=96, freq="15min", tz=TZ)
    # hourly series = mean of its four quarters, as ENTSO-E resampling does today
    quarters = pd.DataFrame({"Solar": np.random.default_rng(1).uniform(0, 100, 96)}, index=quarter_index)
    hourly = quarters.resample("1h").mean()
    assert hourly.index.equals(hourly_index)
    blocks = pd.Series(pd.date_range("2026-01-01 00:00", periods=6, freq="4h", tz=TZ))
    pd.testing.assert_frame_equal(join_block_mean(quarters, blocks, 4), join_block_mean(hourly, blocks, 4))


@pytest.mark.parametrize("start", ["2025-10-25 22:00", "2026-03-28 22:00"])  # autumn and spring clock changes
def test_quarter_hour_fill_survives_clock_change_days(start):
    hourly = _hourly(list(np.arange(10, dtype=float)), start)
    out = hourly_to_quarter_hours(hourly)["temperature_2m_c"]
    assert len(out) == 9 * 4 + 1 and out.notna().all()
