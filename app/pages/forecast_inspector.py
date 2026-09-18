"""Forecast Inspector: pick market, model, date — plot realised vs.
median forecast with 50%/80% quantile fan, with a slider to step through
available cutoffs.
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

from data_loader import available_markets, available_models, compute_wide, load_all_results

st.title("Forecast Inspector")

results = load_all_results()
if results.empty:
    st.warning("No backtest results found in `results/`.")
    st.stop()

wide = compute_wide(results)

markets = available_markets(results)
default_market_idx = markets.index(st.session_state["inspector_market"]) if "inspector_market" in st.session_state and st.session_state["inspector_market"] in markets else 0
market = st.selectbox("Market", markets, index=default_market_idx)

models = available_models(results, market)
default_model_idx = models.index(st.session_state["inspector_model"]) if "inspector_model" in st.session_state and st.session_state["inspector_model"] in models else 0
model = st.selectbox("Model", models, index=default_model_idx)

subset = wide[(wide["market"] == market) & (wide["model"] == model)].copy()
subset["target_time"] = pd.to_datetime(subset["target_time"])
subset["delivery_date"] = subset["target_time"].dt.date

available_dates = sorted(subset["delivery_date"].unique())
if not available_dates:
    st.warning("No data for this market/model combination.")
    st.stop()

default_date = st.session_state.get("inspector_date")
default_date_idx = available_dates.index(default_date) if default_date in available_dates else 0
date = st.selectbox("Delivery date", available_dates, index=default_date_idx)

day_rows = subset[subset["delivery_date"] == date].sort_values("target_time")

cutoffs = sorted(day_rows["cutoff_time"].unique())
if len(cutoffs) > 1:
    cutoff_idx = st.slider("Cutoff (step through if this delivery day has more than one)", 0, len(cutoffs) - 1, 0)
    cutoff = cutoffs[cutoff_idx]
    day_rows = day_rows[day_rows["cutoff_time"] == cutoff]
    st.caption(f"cutoff_time = {cutoff}")
elif cutoffs:
    st.caption(f"cutoff_time = {cutoffs[0]}")

fig, ax = plt.subplots(figsize=(10, 4.5))
x = day_rows["target_time"]

has_80 = "q_0.1" in day_rows.columns and day_rows["q_0.1"].notna().any()
has_50 = "q_0.25" in day_rows.columns and day_rows["q_0.25"].notna().any()

if has_80:
    ax.fill_between(x, day_rows["q_0.1"], day_rows["q_0.9"], alpha=0.2, color="tab:blue", label="80% interval")
if has_50:
    ax.fill_between(x, day_rows["q_0.25"], day_rows["q_0.75"], alpha=0.35, color="tab:blue", label="50% interval")
if "q_0.5" in day_rows.columns:
    ax.plot(x, day_rows["q_0.5"], color="tab:blue", linewidth=2, label="median forecast")
ax.plot(x, day_rows["y_true"], color="black", linewidth=2, linestyle="--", marker="o", markersize=3, label="realised")

ax.set_xlabel("Delivery time")
ax.set_ylabel("Price")
ax.set_title(f"{market} / {model} — {date}")
ax.legend(loc="upper left")
fig.autofmt_xdate()
st.pyplot(fig)

st.dataframe(
    day_rows[[c for c in ["target_time", "horizon_step", "y_true", "q_0.05", "q_0.25", "q_0.5", "q_0.75", "q_0.95"] if c in day_rows.columns]]
    .reset_index(drop=True),
    width="stretch",
)
