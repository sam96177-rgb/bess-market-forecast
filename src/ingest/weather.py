"""Weather ingestion via Open-Meteo's *Historical Forecast* API — the
archive of what was forecast at the time, not the observed-actuals archive.
This is the direct fix for the leakage bug in the old notebook pipeline: a
real day-ahead forecaster never had tomorrow's actual weather, only the
forecast issued for it.

Endpoint (confirmed against the live API, no key required):
    https://historical-forecast-api.open-meteo.com/v1/forecast
    ?latitude=..&longitude=..&start_date=..&end_date=..&hourly=..&timezone=..

Representative points
----------------------
A single weather station is not representative of Germany's wind/solar
fleet, which is spread across the country. We use five points, weighted by
*actual* installed capacity (MW) rather than approximation — sourced from
the Bundesnetzagentur's Marktstammdatenregister (MaStR) statistics report
(Tab. 4, "Gesamte Bruttoleistung ... nach Bundesland", data as of end of
August 2026, retrieved 2026-09-17):

    state                solar MW   wind-onshore MW   combined MW
    Bayern (Bavaria)        34,363             2,886        37,249
    Niedersachsen (Lower S.) 11,539           14,426        25,965
    NRW                     15,705             9,815        25,520
    Brandenburg             10,118             9,672        19,790
    Schleswig-Holstein       5,363            10,064        15,427

The weight for each point is its combined solar+wind MW, used by
`get_capacity_weighted_weather` to blend the five points into one national
series (mirroring the single-location shape of the old feature file).

One judgment call: strictly ranked by combined MW, Baden-Württemberg
(18,335 MW) outranks Schleswig-Holstein (15,427 MW) and would displace it
from a top-5. I kept Schleswig-Holstein instead, because Baden-Württemberg
is — like Bavaria — an inland, solar-dominant southern state, so it would
add little weather diversity beyond what Bavaria's point already covers.
Schleswig-Holstein is the only point representing the northern coastal
weather regime (relevant for both onshore wind and the North Sea/Baltic
offshore fleet, which MaStR attributes to the "Ausschließliche
Wirtschaftszone" / federal EEZ rather than to any single state, so it
can't be captured by a per-state weight at all). Flag if you'd rather
strictly rank by MW and swap in Baden-Württemberg.

Offshore wind capacity (10,999 MW nationally per the same report) is a
real gap: it isn't attributable to any Bundesland in MaStR's own
accounting, so it can't be folded into these weights. Lower Saxony and
Schleswig-Holstein's points at least sit close to the North Sea, which is
the best proxy available without a bespoke offshore-farm coordinate list.
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd
import requests

from src.ingest.cache import RAW_DIR, month_bounds, month_starts

logger = logging.getLogger(__name__)

SOURCE_DIR = RAW_DIR / "weather"
BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TIMEZONE = "Europe/Berlin"
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 5

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_100m",
    "shortwave_radiation",
]

# name -> (lat, lon, weight_mw). Weight is combined installed solar +
# wind-onshore capacity (MW) per Bundesland, from Bundesnetzagentur MaStR
# statistics (data as of Aug 2026) — see module docstring for the source
# table and the Baden-Württemberg/Schleswig-Holstein judgment call.
POINTS = {
    "bavaria": (49.02, 12.10, 37_249),
    "lower_saxony": (53.87, 8.70, 25_965),
    "nrw": (51.72, 8.75, 25_520),
    "brandenburg": (51.76, 14.33, 19_790),
    "schleswig_holstein": (54.48, 9.05, 15_427),
}


def _fetch_one_month(point_name: str, lat: float, lon: float, year: int, month: int) -> pd.DataFrame:
    cache_path = SOURCE_DIR / point_name / f"{year}-{month:02d}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    month_start, month_end = month_bounds(year, month)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": month_start.isoformat(),
        "end_date": month_end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }

    last_error: Exception | None = None
    payload = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "%s %s-%02d attempt %d/%d failed: %s",
                point_name, year, month, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    if last_error is not None:
        raise RuntimeError(
            f"{point_name} {year}-{month:02d} failed after {MAX_RETRIES} attempts"
        ) from last_error

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    # Open-Meteo's "timezone=Europe/Berlin" hourly series always returns
    # exactly 24 rows/day (verified against the real API for 2025-10-25..27,
    # PLAN_V5_REVISION.md V5-4 backfill), never 25 -- it does NOT repeat the
    # autumn fall-back hour the way a genuine DST-aware wall clock would, so
    # there is nothing for ambiguous="infer" to infer from (it needs to see
    # both occurrences) and raises outright. ambiguous=False (ASSUMPTION:
    # treat the single 02:00-02:59 row as the later, post-transition/
    # standard-time instant, not the earlier DST one) picks a fixed side
    # instead — a ~1h label uncertainty on a soft weather covariate, not a
    # settlement price, so the exact side is low-stakes.
    # nonexistent="shift_forward": the spring-forward transition skips a
    # wall-clock hour outright (e.g. 2026-03-29 02:00 never happens in
    # Europe/Berlin) — shifting it forward to 03:00 is the standard
    # treatment, rather than raising and losing the whole month (this broke
    # silently on every March until caught against real data).
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(
        TIMEZONE, ambiguous=False, nonexistent="shift_forward"
    )
    df = df.rename(columns={
        "time": "datetime",
        "temperature_2m": "temperature_2m_c",
        "precipitation": "precipitation_mm",
        "rain": "rain_mm",
        "snowfall": "snowfall_cm",
        "cloud_cover": "cloud_cover_pct",
        "wind_speed_10m": "wind_speed_10m_kmh",
        "wind_speed_100m": "wind_speed_100m_kmh",
        "shortwave_radiation": "shortwave_radiation_w_m2",
    })
    df["point"] = point_name

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def get_point_weather(point_name: str, start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, list[str]]:
    lat, lon, _weight = POINTS[point_name]
    frames = []
    failures: list[str] = []
    for year, month in month_starts(start_date, end_date):
        try:
            frames.append(_fetch_one_month(point_name, lat, lon, year, month))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s %s-%02d permanently failed: %s", point_name, year, month, exc)
            failures.append(f"{year}-{month:02d}")
    combined = pd.concat(frames).sort_values("datetime").reset_index(drop=True) if frames else pd.DataFrame()
    return combined, failures


def get_all_points_weather(start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frames = []
    failures_by_point: dict[str, list[str]] = {}
    for point_name in POINTS:
        df, failures = get_point_weather(point_name, start_date, end_date)
        frames.append(df)
        if failures:
            failures_by_point[point_name] = failures
    combined = pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()
    return combined, failures_by_point


def get_capacity_weighted_weather(start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Blends the five points into one national series using the rough
    capacity weights in POINTS, mirroring the single-location shape the old
    feature file used."""
    long_df, failures_by_point = get_all_points_weather(start_date, end_date)
    if long_df.empty:
        return long_df, failures_by_point

    long_df["weight"] = long_df["point"].map(lambda p: POINTS[p][2])
    value_cols = [
        "temperature_2m_c", "precipitation_mm", "rain_mm", "snowfall_cm",
        "cloud_cover_pct", "wind_speed_10m_kmh", "wind_speed_100m_kmh",
        "shortwave_radiation_w_m2",
    ]

    def weighted_mean(group: pd.DataFrame) -> pd.Series:
        w = group["weight"]
        return pd.Series({col: (group[col] * w).sum() / w.sum() for col in value_cols})

    blended = long_df.groupby("datetime").apply(weighted_mean, include_groups=False).reset_index()
    return blended, failures_by_point
