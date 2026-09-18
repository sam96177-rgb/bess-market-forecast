"""Error Analysis: error heatmap (hour x month), error vs. horizon_step,
residual distribution, and a worst-N-days table linking into the Forecast
Inspector.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import available_markets, available_models, compute_wide, load_all_results

st.title("Error Analysis")

results = load_all_results()
if results.empty:
    st.warning("No backtest results found in `results/`.")
    st.stop()

wide = compute_wide(results)

market = st.selectbox("Market", available_markets(results), key="ea_market")
model = st.selectbox("Model", available_models(results, market), key="ea_model")

subset = wide[(wide["market"] == market) & (wide["model"] == model)].copy()
if subset.empty or "q_0.5" not in subset.columns:
    st.warning("No point-forecast data for this market/model combination.")
    st.stop()

subset["target_time"] = pd.to_datetime(subset["target_time"])
subset["error"] = subset["q_0.5"] - subset["y_true"]
subset["abs_error"] = subset["error"].abs()
subset["delivery_date"] = subset["target_time"].dt.date

col1, col2 = st.columns(2)

with col1:
    st.subheader("Error heatmap (hour x month)")
    pivot = subset.pivot_table(index="hour_of_day", columns="month", values="abs_error", aggfunc="mean")
    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour of day")
        fig.colorbar(im, ax=ax, label="MAE")
        st.pyplot(fig)
    else:
        st.info("Not enough distinct months/hours in this backtest range for a heatmap yet.")

with col2:
    st.subheader("Error vs. horizon_step")
    by_horizon = subset.groupby("horizon_step")["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(by_horizon.index, by_horizon.values, marker="o")
    ax.set_xlabel("horizon_step")
    ax.set_ylabel("MAE")
    st.pyplot(fig)

st.subheader("Residual distribution (q_0.5 - y_true)")
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(subset["error"], bins=30, color="tab:blue", alpha=0.8)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Residual")
st.pyplot(fig)

st.subheader("Worst days by mean absolute error")
n = st.slider("Show top N", 3, 20, 10)
by_day = subset.groupby("delivery_date")["abs_error"].mean().sort_values(ascending=False).head(n)
worst_days = by_day.reset_index()
worst_days.columns = ["delivery_date", "mean_abs_error"]

for _, row in worst_days.iterrows():
    c1, c2 = st.columns([4, 1])
    c1.write(f"{row['delivery_date']} — MAE {row['mean_abs_error']:.2f}")
    if c2.button("Inspect", key=f"inspect_{row['delivery_date']}"):
        st.session_state["inspector_market"] = market
        st.session_state["inspector_model"] = model
        st.session_state["inspector_date"] = row["delivery_date"]
        st.switch_page("pages/forecast_inspector.py")
