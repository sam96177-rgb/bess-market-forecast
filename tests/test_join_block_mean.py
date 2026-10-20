"""join_block_mean was vectorised (PLAN_V5_REVISION.md V5-4): these tests
compare it against a naive per-target-time reference loop, independently
implemented here, over cases the vectorised version could get wrong --
NaN gaps, an empty window, and the 1h-lookup case. Since V7 day_ahead is given 15-minute
inputs (PLAN_V7_REVISION.md V7-1), so the 15-min-target-on-hourly-data case below is only
the function's edge behaviour, not day_ahead's real call shape.
"""
import numpy as np
import pandas as pd

from src.features.build import join_block_mean


def _naive_reference(hourly_df: pd.DataFrame, target_times: pd.Series, block_length_hours: float) -> pd.DataFrame:
    hourly_df = hourly_df.sort_index()
    rows = []
    for t in target_times:
        window_end = t + pd.Timedelta(hours=block_length_hours) - pd.Timedelta(seconds=1)
        rows.append(hourly_df.loc[t:window_end].mean())
    return pd.DataFrame(rows, index=target_times.index)


def _hourly_frame(start: str, periods: int, columns=("temperature_2m_c", "wind_speed_10m_kmh")) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1h", tz="Europe/Berlin")
    rng = np.random.default_rng(0)
    data = {col: rng.uniform(-5, 30, periods) for col in columns}
    return pd.DataFrame(data, index=idx)


def test_matches_naive_reference_for_a_4h_block():
    hourly = _hourly_frame("2026-01-01 00:00", 48)
    target_times = pd.Series(pd.date_range("2026-01-01 00:00", periods=12, freq="4h", tz="Europe/Berlin"))
    expected = _naive_reference(hourly, target_times, block_length_hours=4)
    actual = join_block_mean(hourly, target_times, block_length_hours=4)
    pd.testing.assert_frame_equal(actual, expected)


def test_matches_naive_reference_for_hourly_lookup():
    """block_length_hours == 1 reduces to a plain per-hour lookup."""
    hourly = _hourly_frame("2026-01-01 00:00", 48)
    target_times = pd.Series(pd.date_range("2026-01-01 00:00", periods=48, freq="1h", tz="Europe/Berlin"))
    expected = _naive_reference(hourly, target_times, block_length_hours=1)
    actual = join_block_mean(hourly, target_times, block_length_hours=1)
    pd.testing.assert_frame_equal(actual, expected)


def test_matches_naive_reference_for_quarter_hour_targets_mostly_empty_windows():
    """15-min targets against hourly data (day_ahead's call shape before V7) --
    most windows contain zero hourly rows (all-NaN mean), matching the
    naive reference's behavior exactly, not silently zero-filled."""
    hourly = _hourly_frame("2026-01-01 00:00", 24)
    target_times = pd.Series(pd.date_range("2026-01-01 00:00", periods=96, freq="15min", tz="Europe/Berlin"))
    expected = _naive_reference(hourly, target_times, block_length_hours=0.25)
    actual = join_block_mean(hourly, target_times, block_length_hours=0.25)
    pd.testing.assert_frame_equal(actual, expected)


def test_matches_naive_reference_with_nan_gaps_in_source_data():
    hourly = _hourly_frame("2026-01-01 00:00", 24)
    hourly.iloc[1, 0] = np.nan  # gap in only one column, one hour within a 4h block
    hourly.iloc[2, :] = np.nan  # gap in every column for one full hour
    target_times = pd.Series(pd.date_range("2026-01-01 00:00", periods=6, freq="4h", tz="Europe/Berlin"))
    expected = _naive_reference(hourly, target_times, block_length_hours=4)
    actual = join_block_mean(hourly, target_times, block_length_hours=4)
    pd.testing.assert_frame_equal(actual, expected)


def test_target_time_before_source_data_starts_gives_all_nan_row():
    hourly = _hourly_frame("2026-01-01 06:00", 24)
    target_times = pd.Series([pd.Timestamp("2026-01-01 00:00", tz="Europe/Berlin")])
    result = join_block_mean(hourly, target_times, block_length_hours=4)
    assert result.iloc[0].isna().all()


def test_preserves_original_target_times_index_not_positional():
    """target_times can arrive with a non-default index (e.g. sliced from
    a larger cutoffs frame) -- the result must align back to THAT index,
    not 0..n-1."""
    hourly = _hourly_frame("2026-01-01 00:00", 24)
    target_times = pd.Series(
        pd.date_range("2026-01-01 00:00", periods=3, freq="4h", tz="Europe/Berlin"),
        index=[10, 11, 12],
    )
    result = join_block_mean(hourly, target_times, block_length_hours=4)
    assert list(result.index) == [10, 11, 12]
