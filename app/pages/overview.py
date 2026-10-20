"""Overview: the leaderboard answering "which method is applicable where"
at a glance. Reads only results/*.parquet (via app/data_loader.py) —
never touches a model object or re-runs the backtest.
"""
import sys
from pathlib import Path

# st.Page execs each page's code directly rather than running it as a
# script, so sibling-directory imports need sys.path set up explicitly
# here (using __file__, which Streamlit does set correctly under exec()).
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (_APP_DIR, _APP_DIR.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from data_loader import available_markets, compute_metrics, load_all_results

st.title("Overview")

results = load_all_results()

if results.empty:
    st.warning(
        "No backtest results found in `results/`. Run a backtest "
        "(src.backtest.engine.run_backtest + save_run) first."
    )
    st.stop()

st.caption(
    f"{len(results):,} forecast rows across "
    f"{results['market'].nunique()} market(s), {results['model'].nunique()} model(s), "
    f"{results['run_id'].nunique()} run(s)."
)

metrics = compute_metrics(results, group_by=("market", "model"))

st.subheader("Skill score vs. seasonal_naive (RMSE ratio: 1 = perfect, 0 = ties the naive baseline)")

leaderboard = metrics.pivot(index="model", columns="market", values="skill_score_vs_naive")
_PREFERRED_ORDER = ["seasonal_naive", "linear_lags", "lgbm_point"]
# Any model outside the preferred list still shows up, after it, rather than being silently dropped.
model_order = [m for m in _PREFERRED_ORDER if m in leaderboard.index] + sorted(
    m for m in leaderboard.index if m not in _PREFERRED_ORDER
)
leaderboard = leaderboard.reindex(model_order)

styled = leaderboard.style.background_gradient(cmap="RdYlGn", vmin=-0.5, vmax=1.0, axis=None).format("{:.2f}")
st.dataframe(styled, width="stretch")

with st.expander("What am I looking at?"):
    st.markdown(
        "- **Skill score** = `1 - RMSE(model) / RMSE(seasonal_naive)`, computed on the same "
        "market's rows for both — comparable across markets with very different price levels "
        "(EUR/MWh day-ahead vs. EUR/MW capacity), unlike raw RMSE.\n"
        "- `1.0` = perfect forecast, `0.0` = exactly as good as \"same hour last week\", "
        "negative = worse than the naive baseline.\n"
        "- `seasonal_naive`'s own row is always exactly `0.0` by construction."
    )

st.subheader("Point metrics detail")
market_filter = st.multiselect("Market", available_markets(results), default=available_markets(results))
detail = metrics[metrics["market"].isin(market_filter)].sort_values(
    ["market", "skill_score_vs_naive"], ascending=[True, False]
)
show_cols = [
    c for c in ["market", "model", "n", "mae", "rmse", "skill_score_vs_naive", "coverage_50", "coverage_80"]
    if c in detail.columns
]
st.dataframe(detail[show_cols].reset_index(drop=True), width="stretch")
