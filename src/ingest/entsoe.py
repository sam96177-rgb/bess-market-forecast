"""ENTSO-E ingestion: day-ahead prices, and the *day-ahead published
forecasts* for load and wind/solar generation — never the actuals, since
actuals for a given hour are only known after the fact and would leak
information a real forecaster couldn't have had.

Known failure mode (confirmed against entsoe-py's source): `query_day_ahead_
prices` has no internal chunking at all, so a multi-month span is one raw
HTTP request that can 503 on the ENTSO-E side. `query_load_forecast` and
`query_wind_and_solar_forecast` do have an internal `@month_limited`
decorator, but it only catches `NoMatchingDataError` — a 503 on any single
month still aborts the whole multi-month call with nothing cached. That is
why every fetch here is issued one calendar month at a time from this
module, with its own retry/backoff and its own per-month cache file: a
single bad month can never take down the rest of the run, and a re-run only
has to redo the months that are still missing from the cache.

Resolution mismatch: ENTSO-E publishes load and wind/solar forecasts at
15-minute resolution, while day-ahead prices (and every other market in
this system) run hourly or 4-hourly. The raw cache below stores whatever
ENTSO-E actually returned (15-min, untouched) — raw data should stay
immutable and re-derivable. `get_load_forecast` and
`get_wind_and_solar_forecast` resample that to hourly by default before
returning it, using the mean: these are power readings (MW), not
cumulative energy, so an hourly mean approximates average power over the
hour, which is what an hourly feature should represent (summing 4
15-minute MW readings would overstate by 4x and has no physical meaning).
Pass `resample=False` to get the raw 15-min series instead.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

from src.ingest.cache import RAW_DIR, month_bounds, month_starts

logger = logging.getLogger(__name__)

SOURCE_DIR = RAW_DIR / "entsoe"
COUNTRY_CODE = "DE_LU"
TIMEZONE = "Europe/Berlin"
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 5


def _client() -> EntsoePandasClient:
    load_dotenv()
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY not set. Add it to .env at the repo root."
        )
    return EntsoePandasClient(api_key=api_key)


def _fetch_one_month(series_name: str, fetch_fn, year: int, month: int) -> pd.DataFrame:
    cache_path = SOURCE_DIR / series_name / f"{year}-{month:02d}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    month_start, month_end = month_bounds(year, month)
    start = pd.Timestamp(month_start, tz=TIMEZONE)
    end = pd.Timestamp(month_end, tz=TIMEZONE) + pd.Timedelta(days=1)

    last_error: Exception | None = None
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = fetch_fn(start, end)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - entsoe-py raises assorted HTTP/parse errors
            last_error = exc
            logger.warning(
                "%s %s-%02d attempt %d/%d failed: %s",
                series_name, year, month, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    if last_error is not None:
        raise RuntimeError(
            f"{series_name} {year}-{month:02d} failed after {MAX_RETRIES} attempts"
        ) from last_error

    df = result.to_frame(name=series_name) if isinstance(result, pd.Series) else result
    df.index.name = "datetime"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def fetch_series(series_name: str, fetch_fn, start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, list[str]]:
    """Fetch a series one month at a time. Returns (data, failed_months) —
    failed months are logged and skipped, never aborting the rest of the
    run. Re-calling this later will only re-fetch months still missing from
    the cache.
    """
    frames = []
    failures: list[str] = []
    for year, month in month_starts(start_date, end_date):
        try:
            frames.append(_fetch_one_month(series_name, fetch_fn, year, month))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s %s-%02d permanently failed: %s", series_name, year, month, exc)
            failures.append(f"{year}-{month:02d}")
    combined = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    return combined, failures


def get_day_ahead_prices(start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, list[str]]:
    client = _client()
    return fetch_series(
        "da_price_eur_mwh",
        lambda start, end: client.query_day_ahead_prices(COUNTRY_CODE, start=start, end=end),
        start_date, end_date,
    )


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Mean-resample a 15-minute MW series to hourly. See module docstring
    for why mean (not sum) is the correct aggregation for power readings."""
    if df.empty:
        return df
    return df.resample("1h").mean()


def get_load_forecast(
    start_date: dt.date, end_date: dt.date, resample: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Day-ahead published load forecast (process_type A01), not the actual
    realised load. Returned hourly by default (mean of the native 15-min
    values) — pass resample=False for the raw 15-min series."""
    client = _client()
    df, failures = fetch_series(
        "load_forecast_mw",
        lambda start, end: client.query_load_forecast(
            COUNTRY_CODE, start=start, end=end, process_type="A01"
        ),
        start_date, end_date,
    )
    return (resample_to_hourly(df) if resample else df), failures


def get_wind_and_solar_forecast(
    start_date: dt.date, end_date: dt.date, resample: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Day-ahead published wind & solar generation forecast (process_type
    A01), not actual generation. Returned hourly by default (mean of the
    native 15-min values) — pass resample=False for the raw 15-min series."""
    client = _client()
    df, failures = fetch_series(
        "wind_solar_forecast_mw",
        lambda start, end: client.query_wind_and_solar_forecast(
            COUNTRY_CODE, start=start, end=end, psr_type=None, process_type="A01"
        ),
        start_date, end_date,
    )
    return (resample_to_hourly(df) if resample else df), failures


# ENTSO-E PSR type names (entsoe.mappings.PSRTYPE_MAPPINGS) for the three
# renewable types query_generation's psr_type=None also returns alongside
# every other production type (Fossil Gas, Nuclear, ...) -- filtered out
# at fetch time so the per-month cache holds only what PLAN_V6_REVISION.md
# V6-3 actually uses, the same way query_wind_and_solar_forecast's own
# dedicated endpoint never returns the other types in the first place.
_WIND_AND_SOLAR_PSR_NAMES = ["Wind Onshore", "Wind Offshore", "Solar"]


def get_wind_and_solar_actual(
    start_date: dt.date, end_date: dt.date, resample: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Actual (realised) wind & solar generation -- PLAN_V6_REVISION.md V6-3's
    "past input": what a model may see is bounded by
    src/features/availability.py's after_delivery_rule, never this
    function's date range directly, so a cutoff can only use actuals that
    were already measured by then. Returned hourly by default (mean of the
    native resolution) — pass resample=False for the raw series."""
    client = _client()

    def _fetch(start, end):
        df = client.query_generation(COUNTRY_CODE, start=start, end=end, psr_type=None)
        # query_generation returns columns as a (psr_name, flow) MultiIndex
        # whenever a psr type has both production and pumped-storage/
        # consumption flows (confirmed empirically: Wind Onshore and Solar
        # both carry an "Actual Consumption" column alongside "Actual
        # Aggregated"; Wind Offshore has only "Actual Aggregated"). Keep
        # generation only -- "Actual Consumption" is not a feature we want,
        # and a bare `df[["Wind Onshore", ...]]` on a MultiIndex matches
        # EVERY flow under that name, not just generation.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs("Actual Aggregated", axis=1, level=-1)
        return df[[c for c in _WIND_AND_SOLAR_PSR_NAMES if c in df.columns]]

    df, failures = fetch_series("wind_solar_actual_mw", _fetch, start_date, end_date)
    return (resample_to_hourly(df) if resample else df), failures
