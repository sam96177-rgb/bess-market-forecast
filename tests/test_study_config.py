"""PLAN_V5_REVISION.md V5-6: config/study.yaml is the one file both repos
read for what a study is. These tests cover only this repo's loader; the
optimizer repo's own reader has its own tests."""
import datetime as dt

import pytest

from src.config import load_study_config


def test_loads_the_repo_study_config():
    study = load_study_config()
    assert study.name == "v8"
    assert study.history_start == dt.date(2025, 1, 1)
    assert study.min_training_history_days == 365
    assert study.models == ["seasonal_naive", "lgbm_point", "linear_lags", "lear"]
    assert study.test_window["day_ahead"].start == dt.date(2026, 3, 21)
    assert study.test_window["day_ahead"].end == dt.date(2026, 9, 17)
    assert study.max_test_window_end() == dt.date(2026, 9, 17)
    assert study.validation_window["day_ahead"].start == dt.date(2026, 1, 20)
    assert study.validation_window["day_ahead"].end == dt.date(2026, 3, 20)
    assert study.forecast_start("day_ahead") == dt.date(2026, 1, 20)


def test_forecast_start_falls_back_to_test_window_without_a_validation_window(tmp_path):
    no_validation = tmp_path / "study.yaml"
    no_validation.write_text(
        "name: v5\nhistory_start: 2025-01-01\nmin_training_history_days: 365\n"
        "test_window:\n  day_ahead: {start: 2026-03-21, end: 2026-09-17}\n"
        "models: [seasonal_naive]\n",
        encoding="utf-8",
    )
    study = load_study_config(no_validation)
    assert study.validation_window is None
    assert study.forecast_start("day_ahead") == dt.date(2026, 3, 21)


def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "study.yaml"
    bad.write_text("name: v5\nhistory_start: 2025-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_study_config(bad)


def test_window_end_before_start_raises(tmp_path):
    bad = tmp_path / "study.yaml"
    bad.write_text(
        "name: v5\nhistory_start: 2025-01-01\nmin_training_history_days: 365\n"
        "test_window:\n  day_ahead: {start: 2026-03-21, end: 2026-01-01}\n"
        "models: [seasonal_naive]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is before start"):
        load_study_config(bad)
