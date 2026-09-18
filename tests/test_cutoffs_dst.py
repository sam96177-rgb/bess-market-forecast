"""Regression tests for DST handling in generate_cutoffs. Both fixed
values below (day counts and the FCR spillover boundary) were verified
against real data during a live 6-month backtest — not assumed:
- day_ahead's 92/100/96 counts matched the actual number of rows ENTSO-E
  published for 2026-03-29 / 2026-10-25 / an ordinary day.
- fcr's exact boundary timestamps matched the real regelleistung.net FCR
  file for March 2026 byte-for-byte (including the last block's end
  landing at 01:00 the *next* day, not 00:00).
"""
import datetime as dt

import pandas as pd
import pytest

from src.backtest.cutoffs import generate_cutoffs
from src.config import load_markets_config

MARKETS = load_markets_config()
SPRING_FORWARD_2026 = dt.date(2026, 3, 29)
FALL_BACK_2026 = dt.date(2026, 10, 25)
NORMAL_DAY = dt.date(2026, 6, 15)


def _primary_horizon_rows(market_name: str, day: dt.date) -> pd.DataFrame:
    market = MARKETS[market_name]
    df = generate_cutoffs(market, day, day)
    return df[df["horizon_hours"] == min(market.horizons_hours)]


@pytest.mark.parametrize(
    "day,expected_rows",
    [(SPRING_FORWARD_2026, 92), (FALL_BACK_2026, 100), (NORMAL_DAY, 96)],
)
def test_day_ahead_variable_period_count_matches_real_entsoe_day_length(day, expected_rows):
    rows = _primary_horizon_rows("day_ahead", day)
    assert len(rows) == expected_rows
    assert rows["target_window_start"].is_unique


@pytest.mark.parametrize("day", [SPRING_FORWARD_2026, FALL_BACK_2026, NORMAL_DAY])
def test_fcr_fixed_period_count_is_always_six_blocks(day):
    rows = _primary_horizon_rows("fcr", day)
    assert len(rows) == 6
    assert rows["target_window_start"].is_unique


def test_fcr_dst_day_boundaries_match_real_regelleistung_file():
    """Golden values from data/raw/regelleistung/FCR/RESULT_OVERVIEW_..._
    2026-03-01_2026-03-31.xlsx, parsed by src.ingest.regelleistung — the
    real published data for 2026-03-29, block NEGPOS_20_24."""
    rows = _primary_horizon_rows("fcr", SPRING_FORWARD_2026).sort_values("horizon_step")
    first, last = rows.iloc[0], rows.iloc[-1]
    assert first["target_window_start"] == pd.Timestamp("2026-03-29 00:00", tz="Europe/Berlin")
    assert last["target_window_start"] == pd.Timestamp("2026-03-29 21:00", tz="Europe/Berlin")
    assert last["target_window_end"] == pd.Timestamp("2026-03-30 01:00", tz="Europe/Berlin")


def test_no_duplicate_or_reversed_windows_across_a_dst_spanning_month():
    """General invariant, checked across a whole month containing the
    spring-forward transition: every target window must be unique and
    non-empty (start < end), for every market."""
    for market_name, market in MARKETS.items():
        df = generate_cutoffs(market, dt.date(2026, 3, 1), dt.date(2026, 3, 31))
        primary = df[df["horizon_hours"] == min(market.horizons_hours)]
        assert primary["target_window_start"].is_unique, f"{market_name} has duplicate windows in March 2026"
        assert (primary["target_window_end"] > primary["target_window_start"]).all(), (
            f"{market_name} has a zero/negative-length window in March 2026"
        )
