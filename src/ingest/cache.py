"""Shared helpers for ingest modules: monthly chunking and the on-disk raw
cache. Every source caches to data/raw/<source>/ and skips re-downloading a
month whose cache file already exists.
"""
from __future__ import annotations

import calendar
import datetime as dt
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def month_starts(start_date: dt.date, end_date: dt.date):
    """Yield (year, month) for every calendar month overlapping the range."""
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, last_day)
