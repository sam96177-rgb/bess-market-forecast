"""Walk-forward backtest engine: for each (market, model, cutoff), builds
features available at that cutoff (src.features.build), predicts the
market's full delivery window, and appends to one long-format results
table — the single contract everything downstream (evaluation, the UI,
and eventually a battery optimiser) reads from. Nothing downstream ever
touches a model object directly.

Retraining is quarterly (91-day windows by default), not per-cutoff:
refitting on every single cutoff (24/day for day-ahead) would be far too
slow for a laptop and buys little, since a model's inputs barely change
hour to hour. Each retrain uses an EXPANDING training window — all history
available up to that point, not just the prior quarter — built through
exactly the same per-cutoff availability filter test data goes through, so
the model can never train on a feature it won't actually have at
prediction time either.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import uuid
from pathlib import Path

import pandas as pd

from src.backtest.cutoffs import generate_cutoffs
from src.config import MarketConfig
from src.features.build import build_feature_frame
from src.models.registry import get_model

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

RETRAIN_EVERY_DAYS = 91
DEFAULT_BACKTEST_MONTHS = 6
MIN_TRAIN_ROWS = 10


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _build_training_set(
    market: MarketConfig,
    target_series: pd.Series,
    weather: pd.DataFrame | None,
    train_start: dt.date,
    train_end: dt.date,
) -> tuple[pd.DataFrame, pd.Series]:
    """Builds (X, y) from every daily cutoff in [train_start, train_end],
    at the market's primary (shortest) horizon."""
    if train_end < train_start:
        return pd.DataFrame(), pd.Series(dtype=float)

    cutoffs = generate_cutoffs(market, train_start, train_end)
    primary_horizon = min(market.horizons_hours)
    cutoffs = cutoffs[cutoffs["horizon_hours"] == primary_horizon]

    frames = []
    for cutoff_time, group in cutoffs.groupby("cutoff_time"):
        target_times = group["target_window_start"]
        features, _dropped = build_feature_frame(market, cutoff_time, target_times, target_series, weather=weather)
        frames.append(features)
    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.concat(frames).reset_index(drop=True)
    y = X["target_time"].map(target_series.to_dict())
    feature_cols = [c for c in X.columns if c != "target_time"]
    mask = y.notna() & X[feature_cols].notna().any(axis=1)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def run_backtest(
    market: MarketConfig,
    model_name: str,
    target_series: pd.Series,
    weather: pd.DataFrame | None,
    backtest_start: dt.date,
    backtest_end: dt.date,
    retrain_every_days: int = RETRAIN_EVERY_DAYS,
    model_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward backtest for one (market, model) pair over
    [backtest_start, backtest_end] (inclusive, delivery dates). Returns
    (results_long_df, run_metadata_dict) — see module docstring for the
    results schema and src/backtest/engine.py's save_run for persistence.
    """
    model_kwargs = model_kwargs or {}
    run_id = str(uuid.uuid4())
    primary_horizon = min(market.horizons_hours)

    all_cutoffs = generate_cutoffs(market, backtest_start, backtest_end)
    all_cutoffs = all_cutoffs[all_cutoffs["horizon_hours"] == primary_horizon]

    result_chunks: list[pd.DataFrame] = []
    dropped_features_seen: set[str] = set()
    kept_features_seen: set[str] = set()
    quantiles_used: list[str] = []
    n_windows_trained = 0
    n_windows_skipped = 0

    earliest_available = target_series.index.min().date()

    window_start = backtest_start
    while window_start <= backtest_end:
        window_end = min(window_start + dt.timedelta(days=retrain_every_days - 1), backtest_end)
        train_end = window_start - dt.timedelta(days=1)

        X_train, y_train = _build_training_set(market, target_series, weather, earliest_available, train_end)

        if len(X_train) < MIN_TRAIN_ROWS:
            logger.warning(
                "Skipping window %s..%s: only %d training rows available (need >= %d).",
                window_start, window_end, len(X_train), MIN_TRAIN_ROWS,
            )
            n_windows_skipped += 1
            window_start = window_end + dt.timedelta(days=1)
            continue

        model = get_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)
        n_windows_trained += 1

        # Sliced by DELIVERY day (target_window_start), not cutoff_time's
        # own date — a cutoff is dated D-1 (or earlier) relative to the
        # delivery day it forecasts, so filtering by cutoff_time's date
        # against a delivery-day range silently drops the window's first
        # day and leaks in a day beyond its end.
        window_cutoffs = all_cutoffs[
            (all_cutoffs["target_window_start"].dt.date >= window_start)
            & (all_cutoffs["target_window_start"].dt.date <= window_end)
        ]

        for cutoff_time, group in window_cutoffs.groupby("cutoff_time"):
            target_times = group["target_window_start"]
            horizon_by_target = group.set_index("target_window_start")["horizon_step"]

            X_test, dropped = build_feature_frame(market, cutoff_time, target_times, target_series, weather=weather)
            dropped_features_seen.update(dropped)
            kept_features_seen.update(c for c in X_test.columns if c != "target_time")

            quantiles = getattr(model, "quantiles", None) if model.supports_quantiles else None
            preds = model.predict(X_test, quantiles=quantiles)
            quantiles_used = list(preds.columns)

            # NB: must not go through .values here — for a tz-aware Series
            # it silently drops the tzinfo, which then breaks every
            # downstream .map() lookup keyed by tz-aware timestamps
            # (horizon_step, y_true) without raising, just producing NaN.
            chunk = preds.copy()
            chunk["target_time"] = X_test["target_time"]
            chunk = chunk.melt(id_vars="target_time", var_name="quantile", value_name="y_pred")
            chunk["quantile"] = chunk["quantile"].astype(float)
            chunk["horizon_step"] = chunk["target_time"].map(horizon_by_target)
            chunk["y_true"] = chunk["target_time"].map(target_series.to_dict())
            chunk["cutoff_time"] = cutoff_time
            chunk["market"] = market.name
            chunk["model"] = model_name
            chunk["run_id"] = run_id
            result_chunks.append(chunk)

        window_start = window_end + dt.timedelta(days=1)

    columns = ["market", "model", "run_id", "cutoff_time", "target_time", "horizon_step", "quantile", "y_pred", "y_true"]
    results = pd.concat(result_chunks, ignore_index=True)[columns] if result_chunks else pd.DataFrame(columns=columns)

    metadata = {
        "run_id": run_id,
        "market": market.name,
        "model": model_name,
        "model_kwargs": model_kwargs,
        "backtest_start": str(backtest_start),
        "backtest_end": str(backtest_end),
        "retrain_every_days": retrain_every_days,
        "n_windows_trained": n_windows_trained,
        "n_windows_skipped": n_windows_skipped,
        "quantiles": quantiles_used,
        "kept_features": sorted(kept_features_seen),
        "dropped_features": sorted(dropped_features_seen),
        "data_version": {
            "target_series_start": str(target_series.index.min()),
            "target_series_end": str(target_series.index.max()),
            "n_target_rows": len(target_series),
        },
        "git_commit": _git_commit(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return results, metadata


def default_backtest_range(target_series: pd.Series, months: int = DEFAULT_BACKTEST_MONTHS) -> tuple[dt.date, dt.date]:
    """Rolling-origin default: the most recent `months` of available data."""
    end_date = target_series.index.max().date()
    start_date = end_date - dt.timedelta(days=months * 30)
    return max(start_date, target_series.index.min().date()), end_date


def save_run(results: pd.DataFrame, metadata: dict, results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = metadata["run_id"]
    parquet_path = results_dir / f"{run_id}.parquet"
    json_path = results_dir / f"{run_id}.json"
    results.to_parquet(parquet_path)
    json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return parquet_path, json_path
