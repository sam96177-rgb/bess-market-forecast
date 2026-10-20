"""Regression test for a real bug hit during the V5-2 backfill
(PLAN_V5_REVISION.md): Open-Meteo's "timezone=Europe/Berlin" hourly series
returns exactly 24 rows for every calendar day, including the autumn
DST fall-back day -- it never repeats the ambiguous 02:00-02:59 hour the
way a genuine DST-aware wall clock would (verified against the real API
for 2025-10-25..27). `ambiguous="infer"` requires that repeat to resolve
the ambiguity and raised outright without it, losing the whole month.
"""
import calendar
import datetime as dt

import pandas as pd
import pytest

import src.ingest.weather as weather


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _naive_hourly_payload(year: int, month: int) -> dict:
    """Mirrors the real API: exactly 24 naive 'YYYY-MM-DDTHH:MM' labels
    per calendar day, no repeats on the DST transition day."""
    n_days = calendar.monthrange(year, month)[1]
    times = [
        f"{year}-{month:02d}-{day:02d}T{hour:02d}:00"
        for day in range(1, n_days + 1)
        for hour in range(24)
    ]
    return {"hourly": {"time": times, "temperature_2m": [10.0] * len(times)}}


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "SOURCE_DIR", tmp_path / "weather")


def test_fetch_one_month_does_not_raise_on_the_autumn_dst_fall_back_month(monkeypatch):
    """2025-10 contains 2025-10-26, Europe/Berlin's fall-back day."""
    monkeypatch.setattr(
        weather.requests, "get",
        lambda *a, **kw: _FakeResponse(_naive_hourly_payload(2025, 10)),
    )
    df = weather._fetch_one_month("bavaria", 49.02, 12.10, 2025, 10)

    day = df[df["datetime"].dt.date == dt.date(2025, 10, 26)]
    assert len(day) == 24, "the fall-back day's naive 24-row grid must survive whole, not be dropped or raise"
    assert str(day["datetime"].dt.tz) == "Europe/Berlin"


def test_fetch_one_month_shifts_forward_on_the_spring_dst_gap_month(monkeypatch):
    """2026-03 contains 2026-03-29, Europe/Berlin's spring-forward day."""
    monkeypatch.setattr(
        weather.requests, "get",
        lambda *a, **kw: _FakeResponse(_naive_hourly_payload(2026, 3)),
    )
    df = weather._fetch_one_month("bavaria", 49.02, 12.10, 2026, 3)

    day = df[df["datetime"].dt.date == dt.date(2026, 3, 29)]
    assert len(day) == 24
    assert day["datetime"].is_monotonic_increasing


def test_ambiguous_fall_back_hour_resolves_to_a_fixed_deterministic_offset(monkeypatch):
    monkeypatch.setattr(
        weather.requests, "get",
        lambda *a, **kw: _FakeResponse(_naive_hourly_payload(2025, 10)),
    )
    df = weather._fetch_one_month("bavaria", 49.02, 12.10, 2025, 10)

    ambiguous_row = df[
        (df["datetime"].dt.date == dt.date(2025, 10, 26)) & (df["datetime"].dt.hour == 2)
    ]
    assert len(ambiguous_row) == 1
    # ambiguous=False: the later, post-transition/standard-time (CET, +01:00) instant.
    assert ambiguous_row["datetime"].iloc[0].utcoffset() == pd.Timedelta(hours=1)
