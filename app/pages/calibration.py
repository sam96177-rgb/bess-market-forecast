"""Calibration: reliability diagram (nominal vs. empirical coverage),
pinball loss by quantile, and interval width over time. This is where a
model can look great on RMSE and still be dangerous for a bidding
decision if its intervals are miscalibrated.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_loader import available_markets, available_models, compute_metrics, compute_wide, load_all_results
from src.evaluation.metrics import ALL_QUANTILES, interval_coverage

st.title("Calibration")

results = load_all_results()
if results.empty:
    st.warning("No backtest results found in `results/`.")
    st.stop()

wide = compute_wide(results)

market = st.selectbox("Market", available_markets(results), key="cal_market")


def _has_quantiles(m: str) -> bool:
    rows = wide[(wide["market"] == market) & (wide["model"] == m)]
    return "q_0.05" in rows.columns and rows["q_0.05"].notna().any()


quantile_models = [m for m in available_models(results, market) if _has_quantiles(m)]
if not quantile_models:
    st.warning("No quantile-producing model has results for this market yet.")
    st.stop()
model = st.selectbox("Model", quantile_models, key="cal_model")

subset = wide[(wide["market"] == market) & (wide["model"] == model)].copy()
subset["target_time"] = pd.to_datetime(subset["target_time"])
y_true = subset["y_true"].values

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reliability diagram")
    nominal_pairs = [(0.5, "q_0.25", "q_0.75"), (0.8, "q_0.1", "q_0.9"), (0.9, "q_0.05", "q_0.95")]
    nominal, empirical = [], []
    for level, lo_col, hi_col in nominal_pairs:
        if lo_col in subset.columns and hi_col in subset.columns:
            nominal.append(level)
            empirical.append(interval_coverage(y_true, subset[lo_col].values, subset[hi_col].values))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.plot(nominal, empirical, marker="o", color="tab:red", label="this model")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)

    for level, emp in zip(nominal, empirical):
        flag = "✓" if abs(emp - level) < 0.05 else "⚠️ miscalibrated"
        st.write(f"{int(level*100)}% interval: empirical coverage = {emp:.1%}  {flag}")

with col2:
    st.subheader("Pinball loss by quantile")
    row_metrics = compute_metrics(results, group_by=("market", "model"))
    row = row_metrics[(row_metrics["market"] == market) & (row_metrics["model"] == model)]
    if not row.empty and f"pinball_q{ALL_QUANTILES[0]}" in row.columns:
        losses = [row.iloc[0][f"pinball_q{q}"] for q in ALL_QUANTILES]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(ALL_QUANTILES, losses, marker="o")
        ax.set_xlabel("Quantile")
        ax.set_ylabel("Mean pinball loss")
        st.pyplot(fig)
    else:
        st.info("No pinball loss available for this model.")

st.subheader("Interval width over time (80% interval)")
if "q_0.1" in subset.columns and "q_0.9" in subset.columns:
    subset["width_80"] = subset["q_0.9"] - subset["q_0.1"]
    ts = subset.sort_values("target_time")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts["target_time"], ts["width_80"], color="tab:purple")
    ax.set_xlabel("Delivery time")
    ax.set_ylabel("80% interval width")
    fig.autofmt_xdate()
    st.pyplot(fig)
