"""regelleistung.net ingestion: FCR and aFRR capacity tender results.

The endpoint format below was confirmed by inspecting the live Data Center
UI (https://www.regelleistung.net/apps/datacenter/tendering-files/) rather
than assumed from memory: it serves one static file per calendar month, not
a parameterised date-range API —

    https://www.regelleistung.net/apps/crds/api/v2/tenders/files/
        RESULT_OVERVIEW_CAPACITY_MARKET_<PRODUCT>_<month-start>_<month-end>.xlsx

`<PRODUCT>` is `FCR` or `aFRR`; dates are the first/last calendar day of the
month, e.g. `2024-01-01_2024-01-31`. This is a public, unauthenticated GET
(verified with a direct download outside the browser).

We use the RESULT_OVERVIEW file (aggregated settlement result) rather than
RESULT_LIST_ANONYM (the anonymised per-bid list): RESULT_OVERVIEW is what
your existing merged data was built from — verified exactly, since
GERMANY_AVERAGE_CAPACITY_PRICE for aFRR POS_00_04 / NEG_00_04 on 2024-01-01
is 4.72 / 21.33, identical to processed_features_2024.csv.

FCR settlement price comes from GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW],
filtered to TENDER_NUMBER == 1 (the primary tender; TENDER_NUMBER 2 is an
"additional" round that is almost always empty placeholder rows — confirmed
against a real month where all 6 such rows were "-").

Manual fallback: if the endpoint is unavailable, drop identically-named
xlsx files into data/raw/regelleistung/<product>/ yourself. Parsing reads
whatever is already in that cache directory, regardless of how it got
there — the download step is just what populates it automatically.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.ingest.cache import RAW_DIR, month_bounds, month_starts

logger = logging.getLogger(__name__)

SOURCE_DIR = RAW_DIR / "regelleistung"
BASE_URL = "https://www.regelleistung.net/apps/crds/api/v2/tenders/files"
TIMEZONE = "Europe/Berlin"
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 5

BLOCK_RE = re.compile(r"(\d{2})_(\d{2})$")


def _month_filename(product: str, month_start: dt.date, month_end: dt.date) -> str:
    return (
        f"RESULT_OVERVIEW_CAPACITY_MARKET_{product}_"
        f"{month_start.isoformat()}_{month_end.isoformat()}.xlsx"
    )


def _ensure_month_file(product: str, year: int, month: int) -> Path:
    month_start, month_end = month_bounds(year, month)
    filename = _month_filename(product, month_start, month_end)
    cache_path = SOURCE_DIR / product / filename
    if cache_path.exists():
        return cache_path

    url = f"{BASE_URL}/{filename}"
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(response.content)
            return cache_path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "%s %s-%02d attempt %d/%d failed: %s",
                product, year, month, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(
        f"{product} {year}-{month:02d} failed after {MAX_RETRIES} attempts. "
        f"Manual fallback: download {url} yourself and save it to {cache_path}."
    ) from last_error


def _block_bounds(date_from: dt.date, block_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    match = BLOCK_RE.search(block_name)
    if not match:
        raise ValueError(f"unrecognised block name: {block_name}")
    start_hour, end_hour = int(match.group(1)), int(match.group(2))
    start = pd.Timestamp(date_from, tz=TIMEZONE) + pd.Timedelta(hours=start_hour)
    end_hours = 24 if end_hour == 0 else end_hour
    end = pd.Timestamp(date_from, tz=TIMEZONE) + pd.Timedelta(hours=end_hours)
    return start, end


def _parse_fcr_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[df["TENDER_NUMBER"] == 1].copy()
    price_col = "GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"
    df = df[df[price_col] != "-"]

    bounds = df.apply(
        lambda row: _block_bounds(row["DATE_FROM"].date(), row["PRODUCTNAME"]), axis=1
    )
    df["delivery_start"] = [b[0] for b in bounds]
    df["delivery_end"] = [b[1] for b in bounds]
    df["fcr_price_eur_mw"] = df[price_col].astype(float)
    return df[["delivery_start", "delivery_end", "fcr_price_eur_mw"]].sort_values("delivery_start")


def _parse_afrr_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    price_col = "GERMANY_AVERAGE_CAPACITY_PRICE_[(EUR/MW)/h]"
    df = df[df[price_col] != "-"].copy()

    df["direction"] = df["PRODUCT"].str.split("_").str[0].str.lower()  # "pos" / "neg"

    bounds = df.apply(
        lambda row: _block_bounds(row["DATE_FROM"].date(), row["PRODUCT"]), axis=1
    )
    df["delivery_start"] = [b[0] for b in bounds]
    df["delivery_end"] = [b[1] for b in bounds]
    df["afrr_capacity_eur_mw"] = df[price_col].astype(float)
    return df[["delivery_start", "delivery_end", "direction", "afrr_capacity_eur_mw"]].sort_values(
        "delivery_start"
    )


def _fetch_series(product: str, parse_fn, start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    failures: list[str] = []
    for year, month in month_starts(start_date, end_date):
        try:
            path = _ensure_month_file(product, year, month)
            frames.append(parse_fn(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("%s %s-%02d permanently failed: %s", product, year, month, exc)
            failures.append(f"{year}-{month:02d}")
    combined = pd.concat(frames).reset_index(drop=True) if frames else pd.DataFrame()
    return combined, failures


def get_fcr_capacity_prices(start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, list[str]]:
    return _fetch_series("FCR", _parse_fcr_file, start_date, end_date)


def get_afrr_capacity_prices(start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, list[str]]:
    """Returns both directions in one long-format frame (see `direction`
    column) since they come from the same source file."""
    return _fetch_series("aFRR", _parse_afrr_file, start_date, end_date)
