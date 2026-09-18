"""Hand-verified tests for src.evaluation.resample.expand_results_to_15min
— the output-only step that duplicates FCR/aFRR blocks onto day-ahead's
native 15-min grid for a combined dataset export."""
import pandas as pd

from src.config import load_markets_config
from src.evaluation.resample import expand_results_to_15min

MARKETS = load_markets_config()


def test_day_ahead_passes_through_unchanged():
    """day_ahead is already native 15-min (product_length_hours=0.25) —
    n_steps would be 1, so expansion must be a no-op."""
    results = pd.DataFrame({
        "market": ["day_ahead"], "model": ["seasonal_naive"], "run_id": ["r1"],
        "cutoff_time": [pd.Timestamp("2026-03-01 11:00", tz="Europe/Berlin")],
        "target_time": [pd.Timestamp("2026-03-02 00:00", tz="Europe/Berlin")],
        "horizon_step": [1], "quantile": [0.5], "y_pred": [42.0], "y_true": [40.0],
    })
    expanded = expand_results_to_15min(results, MARKETS["day_ahead"])
    pd.testing.assert_frame_equal(expanded, results)


def test_fcr_block_expands_into_sixteen_15min_rows():
    """FCR's 4h block (product_length_hours=4) must expand into exactly
    4 / 0.25 = 16 rows, one per 15-min sub-interval, spanning the full
    block with no gap and no overlap."""
    block_start = pd.Timestamp("2026-03-02 00:00", tz="Europe/Berlin")
    results = pd.DataFrame({
        "market": ["fcr"], "model": ["seasonal_naive"], "run_id": ["r1"],
        "cutoff_time": [pd.Timestamp("2026-03-01 07:00", tz="Europe/Berlin")],
        "target_time": [block_start],
        "horizon_step": [1], "quantile": [0.5], "y_pred": [12.3], "y_true": [11.9],
    })
    expanded = expand_results_to_15min(results, MARKETS["fcr"])

    assert len(expanded) == 16
    expected_times = pd.date_range(block_start, periods=16, freq="15min")
    assert list(expanded["target_time"]) == list(expected_times)


def test_expanded_rows_duplicate_the_source_value_exactly_not_interpolated():
    """The point of duplication (vs. interpolation) is to never fabricate
    a value the model didn't actually produce — every sub-interval must
    carry the exact same y_pred/y_true as its source block."""
    results = pd.DataFrame({
        "market": ["afrr_pos"], "model": ["lgbm_point"], "run_id": ["r1"],
        "cutoff_time": [pd.Timestamp("2026-03-01 08:00", tz="Europe/Berlin")],
        "target_time": [pd.Timestamp("2026-03-02 04:00", tz="Europe/Berlin")],
        "horizon_step": [1], "quantile": [0.5], "y_pred": [7.77], "y_true": [8.01],
    })
    expanded = expand_results_to_15min(results, MARKETS["afrr_pos"])

    assert (expanded["y_pred"] == 7.77).all()
    assert (expanded["y_true"] == 8.01).all()
    assert (expanded["cutoff_time"] == results["cutoff_time"].iloc[0]).all()


def test_multiple_blocks_each_expand_independently_without_collision():
    """Two adjacent FCR blocks must each expand to their own 16 rows with
    no duplicate target_time values across the combined output."""
    results = pd.DataFrame({
        "market": ["fcr", "fcr"], "model": ["seasonal_naive", "seasonal_naive"], "run_id": ["r1", "r1"],
        "cutoff_time": [pd.Timestamp("2026-03-01 07:00", tz="Europe/Berlin")] * 2,
        "target_time": [
            pd.Timestamp("2026-03-02 00:00", tz="Europe/Berlin"),
            pd.Timestamp("2026-03-02 04:00", tz="Europe/Berlin"),
        ],
        "horizon_step": [1, 2], "quantile": [0.5, 0.5], "y_pred": [1.0, 2.0], "y_true": [1.1, 2.1],
    })
    expanded = expand_results_to_15min(results, MARKETS["fcr"])

    assert len(expanded) == 32
    assert expanded["target_time"].is_unique
