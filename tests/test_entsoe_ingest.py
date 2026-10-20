"""Regression test for a real bug hit during the V6-3 backfill: ENTSO-E's
query_generation(psr_type=None) returns columns as a (psr_name, flow)
MultiIndex whenever a type has more than one flow (confirmed against the
real API: Wind Onshore and Solar both carry "Actual Consumption" alongside
"Actual Aggregated"; Wind Offshore has only "Actual Aggregated"). A bare
df[["Wind Onshore", ...]] selection on that MultiIndex matched every flow
under each name, not just generation -- e.g. df["Wind Onshore"] came back
as a 2-column DataFrame, not a Series, and broke the first real downstream
DataFrame construction that expected one column per PSR type.
"""
import datetime as dt

import pandas as pd
import pytest

import src.ingest.entsoe as entsoe


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(entsoe, "SOURCE_DIR", tmp_path / "entsoe")


class _FakeClient:
    def __init__(self, generation_df: pd.DataFrame):
        self._generation_df = generation_df

    def query_generation(self, country_code, start, end, psr_type=None, **kwargs):
        return self._generation_df


def _multiindex_generation_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=3, freq="1h", tz="Europe/Berlin")
    columns = pd.MultiIndex.from_tuples([
        ("Wind Onshore", "Actual Consumption"),
        ("Wind Onshore", "Actual Aggregated"),
        ("Wind Offshore", "Actual Aggregated"),  # no Consumption flow for this one
        ("Solar", "Actual Consumption"),
        ("Solar", "Actual Aggregated"),
        ("Fossil Gas", "Actual Aggregated"),  # must be dropped -- not wind/solar
    ])
    data = [[1.0, 10.0, 20.0, 2.0, 30.0, 999.0]] * 3
    return pd.DataFrame(data, index=idx, columns=columns)


def test_get_wind_and_solar_actual_keeps_only_generation_one_column_per_type(monkeypatch):
    monkeypatch.setattr(entsoe, "_client", lambda: _FakeClient(_multiindex_generation_frame()))

    df, failures = entsoe.get_wind_and_solar_actual(dt.date(2025, 1, 1), dt.date(2025, 1, 1), resample=False)

    assert failures == []
    assert not isinstance(df.columns, pd.MultiIndex)
    assert sorted(df.columns) == ["Solar", "Wind Offshore", "Wind Onshore"]
    assert df["Wind Onshore"].ndim == 1
    assert (df["Wind Onshore"] == 10.0).all()  # Actual Aggregated, not Actual Consumption (1.0)
    assert (df["Solar"] == 30.0).all()
    assert (df["Wind Offshore"] == 20.0).all()


def test_get_wind_and_solar_actual_handles_a_flat_index_unchanged(monkeypatch):
    """Not every month necessarily comes back with a MultiIndex -- a flat
    column layout (e.g. if every type had exactly one flow) must pass
    through unchanged, not raise on `.xs`."""
    idx = pd.date_range("2025-01-01", periods=2, freq="1h", tz="Europe/Berlin")
    flat_df = pd.DataFrame({"Wind Onshore": [1.0, 2.0], "Solar": [3.0, 4.0]}, index=idx)
    monkeypatch.setattr(entsoe, "_client", lambda: _FakeClient(flat_df))

    df, failures = entsoe.get_wind_and_solar_actual(dt.date(2025, 1, 1), dt.date(2025, 1, 1), resample=False)

    assert failures == []
    assert sorted(df.columns) == ["Solar", "Wind Onshore"]
