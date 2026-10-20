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


FeatureCache = dict[pd.Timestamp, tuple[pd.DataFrame, list]]


def _cached_feature_frame(
    cache: FeatureCache,
    market: MarketConfig,
    cutoff_time: pd.Timestamp,
    target_times: pd.Series,
    target_series: pd.Series,
    weather: pd.DataFrame | None,
    generation_forecast: pd.DataFrame | None = None,
    generation_actual: pd.DataFrame | None = None,
    generation_forecast_published_local: dt.time | None = None,
) -> tuple[pd.DataFrame, list]:
    """Memoizes build_feature_frame by cutoff_time (PLAN_V5_REVISION.md
    V5-4): a cutoff's feature rows and its availability-filter drops
    depend only on that cutoff and the market (src.backtest.cutoffs.
    generate_cutoffs produces the identical target group for a cutoff
    regardless of the requested date range) -- never on whether it's
    serving a retrain's expanding training window or a test window. So
    within one cache's lifetime a given cutoff is built at most once,
    instead of once per retrain per model.
    """
    if cutoff_time not in cache:
        cache[cutoff_time] = build_feature_frame(
            market, cutoff_time, target_times, target_series, weather=weather,
            generation_forecast=generation_forecast, generation_actual=generation_actual,
            generation_forecast_published_local=generation_forecast_published_local,
        )
    return cache[cutoff_time]


def _build_training_set(
    market: MarketConfig,
    target_series: pd.Series,
    weather: pd.DataFrame | None,
    train_start: dt.date,
    train_end: dt.date,
    feature_cache: FeatureCache,
    generation_forecast: pd.DataFrame | None = None,
    generation_actual: pd.DataFrame | None = None,
    generation_forecast_published_local: dt.time | None = None,
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
        features, _dropped = _cached_feature_frame(
            feature_cache, market, cutoff_time, target_times, target_series, weather,
            generation_forecast, generation_actual, generation_forecast_published_local,
        )
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
    min_training_history_days: int | None = None,
    feature_cache: FeatureCache | None = None,
    generation_forecast: pd.DataFrame | None = None,
    generation_actual: pd.DataFrame | None = None,
    generation_forecast_published_local: dt.time | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward backtest for one (market, model) pair over
    [backtest_start, backtest_end] (inclusive, delivery dates). Returns
    (results_long_df, run_metadata_dict) — see module docstring for the
    results schema and src/backtest/engine.py's save_run for persistence.

    `min_training_history_days` (PLAN_V5_REVISION.md V5-2): if given and
    the first retrain window's training history is shorter than this, a
    loud warning is logged. `first_train_days` is always recorded in the
    metadata, guard or not.

    `feature_cache` (PLAN_V5_REVISION.md V5-4): pass one in to share it
    across several run_backtest calls for the SAME market (e.g. one per
    model in scripts/run_backtest.py's loop) so a cutoff's feature frame
    is built once total, not once per model. Defaults to a fresh cache
    scoped to just this call.

    `generation_forecast`/`generation_actual`/`generation_forecast_published_local`
    (PLAN_V6_REVISION.md V6-3): see src.features.build.add_generation_features.
    """
    model_kwargs = model_kwargs or {}
    feature_cache = {} if feature_cache is None else feature_cache
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
    model_diagnostics: list[dict] = []

    earliest_available = target_series.index.min().date()
    first_train_days = (backtest_start - earliest_available).days
    if min_training_history_days is not None and first_train_days < min_training_history_days:
        logger.warning(
            "%s/%s: first training window has only %d days of history before %s "
            "(need >= %d; earliest available data point is %s).",
            market.name, model_name, first_train_days, backtest_start, min_training_history_days, earliest_available,
        )

    window_start = backtest_start
    while window_start <= backtest_end:
        window_end = min(window_start + dt.timedelta(days=retrain_every_days - 1), backtest_end)
        train_end = window_start - dt.timedelta(days=1)

        X_train, y_train = _build_training_set(
            market, target_series, weather, earliest_available, train_end, feature_cache,
            generation_forecast, generation_actual, generation_forecast_published_local,
        )

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
        if hasattr(model, "diagnostics"):
            model_diagnostics.append({"window_start": str(window_start), **model.diagnostics()})

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

            X_test, dropped = _cached_feature_frame(
                feature_cache, market, cutoff_time, target_times, target_series, weather,
                generation_forecast, generation_actual, generation_forecast_published_local,
            )
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
        "first_train_days": first_train_days,
        "quantiles": quantiles_used,
        "model_diagnostics": model_diagnostics,
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
