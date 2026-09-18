"""Shared, cached data loading for the Streamlit app. Reads only
results/*.parquet — nothing here touches a model object or re-runs any
part of the backtest. Every page imports from here so the parquet files
and the metrics table are each computed once per session, not once per
widget interaction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` puts app/ on sys.path, not the repo
# root — add the root so `src.*` imports resolve regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.metrics import evaluate_results, pivot_to_wide  # noqa: E402

RESULTS_DIR = _REPO_ROOT / "results"

RESULTS_COLUMNS = [
    "market", "model", "run_id", "cutoff_time", "target_time",
    "horizon_step", "quantile", "y_pred", "y_true",
]


@st.cache_data
def load_all_results() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=RESULTS_COLUMNS)
    return pd.concat(frames, ignore_index=True)


@st.cache_data
def compute_wide(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    return pivot_to_wide(results)


@st.cache_data
def compute_metrics(results: pd.DataFrame, group_by: tuple[str, ...]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return evaluate_results(results, group_by=list(group_by))


def available_markets(results: pd.DataFrame) -> list[str]:
    return sorted(results["market"].unique()) if not results.empty else []


def available_models(results: pd.DataFrame, market: str | None = None) -> list[str]:
    df = results if market is None else results[results["market"] == market]
    return sorted(df["model"].unique()) if not df.empty else []
