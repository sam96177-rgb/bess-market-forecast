# BESS Market Forecast — Status & Next-Step Planning

Handoff doc for continuing planning in a fresh chat. Covers what's built and
verified so far, and the new direction: a battery optimizer + a combined
forecaster/optimizer dashboard.

## 1. What this project is

A German BESS (battery energy storage system) market forecasting pipeline.
Forecasts 4 markets: day-ahead energy price, FCR capacity price, aFRR+ and
aFRR- capacity prices. Tree models and statistical baselines only (no
deep learning). Repo: `bess-market-forecast`, one git history, config-driven
architecture (`config/markets.yaml` is the single source of truth for all
market timing/product constants — nothing market-specific is hardcoded in
code).

## 2. What's built and verified (Milestones 0–7, complete)

- **Ingestion** (`src/ingest/`): ENTSO-E (day-ahead prices, load/wind/solar
  forecasts), regelleistung.net (FCR/aFRR), Open-Meteo (weather). Real data
  only — the project's hard rule is "never fabricate data; if an API fails
  or a column is missing, stop and say so."
- **Feature availability enforcement** (`src/features/availability.py`):
  every feature is registered with a rule for exactly when it's knowable
  before a cutoff. Unregistered = dropped, fail-closed, logged loudly. This
  is what makes the backtest leakage-free.
- **Walk-forward backtest engine** (`src/backtest/engine.py`): quarterly
  retrain, expanding training window, long-format output
  (`market, model, run_id, cutoff_time, target_time, horizon_step, quantile, y_pred, y_true`).
- **5 models** (`src/models/`): `seasonal_naive`, `linear_lags`,
  `lgbm_point`, and (added in optimizer plan v3) `naive_level_adjusted` and
  `analog_day`. `lgbm_quantile` / `lgbm_quantile_tuned` were removed in v3
  (recoverable from git; Optuna dropped from requirements).
- **Evaluation** (`src/evaluation/metrics.py`): RMSE/MAE, skill score vs.
  naive, pinball loss, interval coverage, Winkler score.
- **Streamlit UI** (`app/`): Overview, Forecast Inspector, Error Analysis,
  Calibration pages — reads directly from `results/*.parquet`.
- **15-min unified export** (`src/evaluation/resample.py`,
  `scripts/export_15min_results.py`, new this session): FCR/aFRR (native
  4h blocks) get duplicated onto day-ahead's native 15-min grid, output-only
  — never fed back into training/evaluation, since the models are and
  should stay scored at native resolution.
- 69 tests passing.

### Real bugs found & fixed getting a trustworthy 6-month backtest
A short test window hid these; the real 6-month range crossed the
2026-03-29 DST transition and exposed:
1. Weather ingestion crashed outright on the DST day.
2. Day-ahead genuinely moved to 15-min products (EU SDAC reform, Oct 2025)
   — config was wrong (hourly).
3. Cutoff generation silently duplicated/miscounted timestamps on the DST
   day.
4. Day-ahead and FCR/aFRR follow **different real DST conventions**
   (verified against each source's actual published files) — now modeled
   explicitly per market.
5. "Same hour last week" lag features used a fixed 168h offset instead of
   calendar-day arithmetic — silently wrong for a week after every DST
   change. This was the one that actually broke real numbers.
6. Fixing #5 with `pd.DateOffset` then crashed on the DST gap itself —
   needed strip/shift/re-localize handling.

Accepted, documented limitations (not bugs): FCR/aFRR have 10 unavoidable
NaN predictions/market (the DST day + its one-week echo); ENTSO-E published
one real day (2026-09-13) at hourly instead of 15-min resolution.

### Final 6-month backtest results (real data, 20 runs, 350,710 rows, zero duplicates)

Skill score vs. `seasonal_naive` (`1 - rmse_model/rmse_naive`; positive = beats naive):

| model | afrr_neg | afrr_pos | day_ahead | fcr |
|---|---|---|---|---|
| seasonal_naive | 0.000 | 0.000 | 0.000 | 0.000 |
| linear_lags | −0.513 | 0.065 | −0.163 | −0.325 |
| lgbm_point | −0.340 | 0.103 | 0.069 | −0.310 |
| naive_level_adjusted | −0.285 | 0.014 | **0.146** | −0.050 |
| analog_day | −0.306 | −0.141 | 0.040 | −0.151 |
| ~~lgbm_quantile~~ (removed) | −0.556 | 0.072 | 0.099 | −0.379 |
| ~~lgbm_quantile_tuned~~ (removed) | −0.509 | 0.058 | 0.124 | −0.406 |

**Key findings**:
- LightGBM models beat naive on day-ahead (best: 0.124) and marginally on
  afrr_pos.
- On FCR and afrr_neg, *every* model is worse than naive — and worse the
  more complex the model. Capacity-price autocorrelation dominates; added
  model complexity actively hurts there.
- Quantile intervals are meaningfully overconfident everywhere (e.g.
  day-ahead's measured 50%/80% coverage: 32%/59% vs. nominal 50%/80%).

## 3. Optimizer: BUILT (historical planning notes below)

> The battery optimizer, the forecaster connection and the combined dashboard
> now exist in the sibling repo `../bess-optimizer` (see its README and
> `PLAN_V3_REVISION.md`). It reads this repo's `results/*.parquet` read-only.
> Nothing in this repo writes anything for it, so the handoff is a convention
> (a relative path), not a contract. The text below is the original plan and
> is kept for context only.

### Original direction

> "Show the result, then build a battery optimizer for trading in
> multi-market. Later connect the forecaster and optimizer, and visualize
> how it works — including a dashboard of the battery and the forecasting
> + decision points the optimizer and forecaster made."

Three pieces, roughly in this order:

### 3a. Battery optimizer (new component)
Given price/capacity forecasts (or realized prices, for a first pass),
decide how to operate a BESS across the 4 markets simultaneously
(day-ahead energy arbitrage + FCR + aFRR+ + aFRR- capacity commitments).

Open questions for the next chat to settle before building:
- **Optimization method**: linear/mixed-integer program (deterministic,
  using point forecasts or realized prices) vs. stochastic optimization
  over the quantile forecasts already produced? Given the "no DL, tree
  models and statistical baselines only" constraint was about the
  *forecaster*, an LP/MILP solver (e.g. PuLP, `linopy`, or `cvxpy`) is the
  natural fit for the *optimizer* — worth confirming that's the intent.
- **Battery model fidelity**: SOC bounds, charge/discharge power limits,
  round-trip efficiency, degradation cost (if any) — how detailed does the
  physical model need to be for v1?
- **Multi-market co-optimization constraints**: can capacity commitments
  (FCR/aFRR) and day-ahead energy trades draw on the same SOC/power
  simultaneously, or are they mutually exclusive per block? This is a real
  German-market rule question (e.g. FCR symmetric reservation vs. aFRR
  reserved headroom) worth getting right before coding.
- **Horizon/rolling structure**: does the optimizer re-solve at each
  market's own gate closure (day-ahead 12:00 D-1, FCR/aFRR ~08:00 D-1),
  mirroring the forecaster's own cutoff structure? That would let it reuse
  `config/markets.yaml`'s existing timing fields directly.

### 3b. Forecaster ↔ optimizer connection
Once the optimizer exists standalone (e.g. tested against realized prices
first), feed it the forecaster's quantile outputs from `results/*.parquet`
(or a live forecast call) instead of realized prices, and compare
optimizer decisions/expected value under forecast vs. actual.

### 3c. Combined dashboard
Extend the existing Streamlit app (`app/`) — or a new page in it — to show:
- Battery state over time (SOC, charge/discharge schedule).
- Which market(s) the optimizer committed capacity/energy to at each
  decision point.
- The forecast (with uncertainty bands) that informed each decision,
  side-by-side with what actually happened.
- Ideally: the realized P&L / value captured vs. a naive or perfect-
  foresight benchmark.

## 4. Relevant existing files to build from

- `config/markets.yaml` — market timing/constants (optimizer should reuse,
  not duplicate).
- `src/backtest/engine.py`, `results/*.parquet` — forecast output schema
  the optimizer will consume.
- `src/evaluation/resample.py` — 15-min unified grid, useful if the
  optimizer wants one common timestep across all 4 markets.
- `app/data_loader.py`, `app/pages/` — existing Streamlit patterns
  (`st.navigation`/`st.Page`, `_bootstrap` sys.path fix) to extend rather
  than reinvent.
- `src/config.py` — `MarketConfig` dataclass / `load_markets_config()`.

## 5. Not yet decided — bring these to the planning chat

1. Optimizer method (LP/MILP vs. stochastic vs. rule-based heuristic).
2. Battery physical parameters (capacity, power rating, efficiency,
   degradation) — real BESS spec or a reasonable assumed one?
3. Whether FCR/aFRR's negative skill scores need addressing (better
   features? accept naive as the production model for those two markets?)
   before wiring the optimizer to them.
4. Whether the optimizer is a new top-level module (`src/optimizer/`) in
   this same repo, or a separate repo that imports this one's forecast
   output (there's already an untracked `market-optimizer/` folder
   sitting in the parent directory — worth clarifying if that's meant to
   be this).
