"""Benchmark table and figure for README.md, computed from the backtest output in results/.

Scores every model on its market's test window (config/study.yaml), on the delivery periods
that every model forecast, against the seasonal_naive baseline. Writes
docs/figures/day_ahead_sample_week.png and prints the table as Markdown.

    python scripts/make_readme_figures.py [--week-start 2026-06-08]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_study_config  # noqa: E402

BASELINE = "seasonal_naive"
TZ = "Europe/Berlin"
MARKET_LABELS = {"day_ahead": "Day-ahead (EUR/MWh)", "fcr": "FCR (EUR/MW)", "afrr_pos": "aFRR+ (EUR/MW)", "afrr_neg": "aFRR- (EUR/MW)"}
# Reference data-viz palette: text ink, then categorical slots 1 and 2.
INK, SERIES_1, SERIES_2, GRID = "#0b0b0b", "#2a78d6", "#eb6834", "#e4e3df"


def load_forecasts(results_dir: Path) -> pd.DataFrame:
    frames = []
    for meta_path in results_dir.glob("*.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(meta_path.with_suffix(".parquet"))
        frames.append(frame.assign(market=meta["market"], model=meta["model"]))
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts["target_time"] = forecasts["target_time"].dt.tz_convert(TZ)
    return forecasts.dropna(subset=["y_pred", "y_true"])


def benchmark(forecasts: pd.DataFrame, study) -> pd.DataFrame:
    """RMSE, MAE and RMSE gain vs BASELINE per market and model, on the test window's shared periods."""
    rows = []
    for market, frame in forecasts.groupby("market"):
        window = study.test_window[market]
        day = frame["target_time"].dt.date
        frame = frame[(day >= window.start) & (day <= window.end)]
        wide = frame.pivot_table(index="target_time", columns="model", values="y_pred").dropna()
        truth = frame.drop_duplicates("target_time").set_index("target_time")["y_true"].reindex(wide.index)
        errors = wide.sub(truth, axis=0)
        rmse = np.sqrt((errors**2).mean())
        for model in wide.columns:
            rows.append({
                "market": market, "model": model, "periods": len(wide),
                "rmse": rmse[model], "mae": errors[model].abs().mean(),
                "rmse_gain_pct": 100 * (1 - rmse[model] / rmse[BASELINE]),
            })
    return pd.DataFrame(rows)


def sample_week_figure(forecasts: pd.DataFrame, week_start: dt.date, path: Path) -> None:
    frame = forecasts[forecasts["market"] == "day_ahead"]
    day = frame["target_time"].dt.date
    frame = frame[(day >= week_start) & (day < week_start + dt.timedelta(days=7))]
    wide = frame.pivot_table(index="target_time", columns="model", values="y_pred")
    truth = frame.drop_duplicates("target_time").set_index("target_time")["y_true"].sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    ax.plot(truth.index, truth.values, color=INK, linewidth=2, label="Real price")
    ax.plot(wide.index, wide["lgbm_point"], color=SERIES_1, linewidth=2, label="lgbm_point forecast")
    ax.plot(wide.index, wide[BASELINE], color=SERIES_2, linewidth=1.5, linestyle="--", label="seasonal_naive (same time last week)")
    ax.set_ylabel("EUR/MWh", color=INK)
    ax.set_title(f"German day-ahead price, week of {week_start:%d %b %Y}: forecasts made at 11:00 the day before", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
    ax.margins(x=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=TZ))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week-start", type=dt.date.fromisoformat, default=dt.date(2026, 6, 8))
    args = parser.parse_args(argv)

    study = load_study_config()
    forecasts = load_forecasts(REPO_ROOT / "results")
    table = benchmark(forecasts, study)
    sample_week_figure(forecasts, args.week_start, REPO_ROOT / "docs" / "figures" / "day_ahead_sample_week.png")

    print("| Market | Model | RMSE | MAE | RMSE vs seasonal_naive |")
    print("|---|---|---|---|---|")
    for market in MARKET_LABELS:
        part = table[table["market"] == market].sort_values("rmse")
        for _, row in part.iterrows():
            gain = "baseline" if row["model"] == BASELINE else f"{row['rmse_gain_pct']:.0f}% lower"
            print(f"| {MARKET_LABELS[market]} | `{row['model']}` | {row['rmse']:.1f} | {row['mae']:.1f} | {gain} |")
    print()
    print(table.groupby("market")["periods"].first().to_string())


if __name__ == "__main__":
    main()
