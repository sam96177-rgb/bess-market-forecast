"""Hand-verified tests for the evaluation metrics — these formulas back
the headline "which model wins where, and is it calibrated" claims, so
each one is checked against a manually computed expected value, not just
"runs without error".
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    interval_coverage,
    pinball_loss,
    winkler_score,
    add_skill_score,
    evaluate,
    evaluate_results,
    pivot_to_wide,
)


def test_pinball_loss_at_median_is_half_mean_absolute_error():
    """Well-known identity: pinball loss at q=0.5 equals half the MAE."""
    y_true = np.array([10.0, 5.0, 20.0, 3.0])
    y_pred = np.array([8.0, 7.0, 25.0, 3.0])
    expected = 0.5 * np.mean(np.abs(y_true - y_pred))
    assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(expected)


def test_pinball_loss_underprediction_at_high_quantile():
    """y_true=10, y_pred=8 (under-prediction), q=0.9: loss = q * (y-pred) = 0.9*2 = 1.8."""
    assert pinball_loss(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)


def test_pinball_loss_overprediction_at_high_quantile():
    """y_true=8, y_pred=10 (over-prediction), q=0.9: loss = (1-q) * (pred-y) = 0.1*2 = 0.2."""
    assert pinball_loss(np.array([8.0]), np.array([10.0]), 0.9) == pytest.approx(0.2)


def test_interval_coverage_exact_fraction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.full(5, 2.0)
    upper = np.full(5, 4.0)
    # only 2, 3, 4 fall inside [2, 4] -> 3/5 = 0.6
    assert interval_coverage(y_true, lower, upper) == pytest.approx(0.6)


def test_winkler_score_hand_computed():
    """80% interval (alpha=0.2), lower=0, upper=10:
    - y=5 (inside): score = width = 10
    - y=-5 (below): score = width + (2/alpha)*(lower - y) = 10 + 10*5 = 60
    - y=15 (above): score = width + (2/alpha)*(y - upper) = 10 + 10*5 = 60
    Mean of [10, 60, 60] = 43.333...
    """
    y_true = np.array([5.0, -5.0, 15.0])
    lower = np.full(3, 0.0)
    upper = np.full(3, 10.0)
    expected = np.mean([10.0, 60.0, 60.0])
    assert winkler_score(y_true, lower, upper, alpha=0.2) == pytest.approx(expected)


def _long_format_fixture() -> pd.DataFrame:
    """Two cutoffs, one target each, one quantile model + one point model
    (seasonal_naive) on the same market, with hand-chosen values so RMSE
    and skill score are easy to check by hand."""
    rows = []
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    targets = [
        (pd.Timestamp("2024-06-03 00:00", tz="Europe/Berlin"), 1, 50.0),
        (pd.Timestamp("2024-06-03 01:00", tz="Europe/Berlin"), 2, 60.0),
    ]
    for target_time, horizon_step, y_true in targets:
        # lgbm_quantile: median predicts exactly y_true (rmse=0), spread +-5 around it
        for q in quantiles:
            spread = (q - 0.5) * 10  # 0 at median, +-5 at the extremes
            rows.append({
                "market": "day_ahead", "model": "lgbm_quantile", "run_id": "r1",
                "cutoff_time": pd.Timestamp("2024-06-02 11:00", tz="Europe/Berlin"),
                "target_time": target_time, "horizon_step": horizon_step,
                "quantile": q, "y_pred": y_true + spread, "y_true": y_true,
            })
        # seasonal_naive: point forecast off by exactly 10
        rows.append({
            "market": "day_ahead", "model": "seasonal_naive", "run_id": "r2",
            "cutoff_time": pd.Timestamp("2024-06-02 11:00", tz="Europe/Berlin"),
            "target_time": target_time, "horizon_step": horizon_step,
            "quantile": 0.5, "y_pred": y_true + 10, "y_true": y_true,
        })
    return pd.DataFrame(rows)


def test_pivot_to_wide_shape_and_derived_columns():
    long_df = _long_format_fixture()
    wide = pivot_to_wide(long_df)
    # 2 targets for lgbm_quantile + 2 for seasonal_naive = 4 rows
    assert len(wide) == 4
    assert "q_0.5" in wide.columns
    assert "hour_of_day" in wide.columns
    naive_row = wide[wide["model"] == "seasonal_naive"].iloc[0]
    assert naive_row["hour_of_day"] in (0, 1)
    assert naive_row["month"] == 6
    assert bool(naive_row["is_weekend"]) == (pd.Timestamp("2024-06-03").dayofweek >= 5)


def test_lgbm_quantile_has_zero_rmse_and_naive_has_rmse_ten():
    long_df = _long_format_fixture()
    metrics = evaluate_results(long_df, group_by=["market", "model"])
    lgbm_row = metrics[metrics["model"] == "lgbm_quantile"].iloc[0]
    naive_row = metrics[metrics["model"] == "seasonal_naive"].iloc[0]
    assert lgbm_row["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert naive_row["rmse"] == pytest.approx(10.0)


def test_skill_score_vs_naive_is_one_when_model_is_perfect():
    long_df = _long_format_fixture()
    metrics = evaluate_results(long_df, group_by=["market", "model"])
    lgbm_row = metrics[metrics["model"] == "lgbm_quantile"].iloc[0]
    assert lgbm_row["skill_score_vs_naive"] == pytest.approx(1.0)


def test_skill_score_vs_naive_is_zero_when_model_equals_baseline():
    long_df = _long_format_fixture()
    # duplicate seasonal_naive's own rows under a fake "identical_model" name
    naive_rows = long_df[long_df["model"] == "seasonal_naive"].copy()
    naive_rows["model"] = "identical_model"
    combined = pd.concat([long_df, naive_rows], ignore_index=True)
    metrics = evaluate_results(combined, group_by=["market", "model"])
    identical_row = metrics[metrics["model"] == "identical_model"].iloc[0]
    assert identical_row["skill_score_vs_naive"] == pytest.approx(0.0)


def test_coverage_100_percent_when_quantiles_hand_constructed_wide():
    """Interval [q_0.1, q_0.9] wraps y_true by construction in the fixture
    (spread of +-4 at those quantiles vs y_true at the median) -> 100% coverage."""
    long_df = _long_format_fixture()
    metrics = evaluate_results(long_df, group_by=["market", "model"])
    lgbm_row = metrics[metrics["model"] == "lgbm_quantile"].iloc[0]
    assert lgbm_row["coverage_80"] == pytest.approx(1.0)
    assert lgbm_row["coverage_50"] == pytest.approx(1.0)
    assert lgbm_row["interval_width_80"] == pytest.approx(8.0)  # q_0.9 - q_0.1 = spread of 4 each side
    assert lgbm_row["interval_width_50"] == pytest.approx(5.0)  # q_0.75 - q_0.25 = spread of 2.5 each side


def test_point_only_model_has_no_probabilistic_columns():
    long_df = _long_format_fixture()
    metrics = evaluate_results(long_df, group_by=["market", "model"])
    naive_row = metrics[metrics["model"] == "seasonal_naive"].iloc[0]
    assert pd.isna(naive_row.get("coverage_80", np.nan))


def test_evaluate_can_slice_by_horizon_step():
    long_df = _long_format_fixture()
    metrics = evaluate_results(long_df, group_by=["market", "model", "horizon_step"])
    assert set(metrics["horizon_step"]) == {1, 2}
    assert len(metrics) == 4  # 2 models x 2 horizon steps
