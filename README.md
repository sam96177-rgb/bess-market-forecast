# bess-market-forecast

German BESS market forecasting system: per-market probabilistic price
forecasts for Day-Ahead (DE-LU), FCR, and aFRR capacity, plus a Streamlit
monitoring UI comparing methods across markets. Tree models and
statistical baselines only — no deep learning.

This build stops at reliable forecasts + monitoring. No battery model, no
dispatch optimisation — see "Explicitly out of scope" below.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash; venv\Scripts\activate.bat on cmd
pip install -r requirements.txt
cp .env.example .env           # then fill in ENTSOE_API_KEY
```

Get an ENTSO-E API key from the [Transparency Platform](https://transparency.entsoe.eu/) (account settings → "Web Api Security Token"). regelleistung.net and Open-Meteo need no key.

## Running

```bash
make ingest      # download & cache raw data (data/raw/, gitignored)
make backtest    # walk-forward backtest, all markets x all models -> results/
make app         # streamlit run app/streamlit_app.py
make all         # all three, in order
make test        # pytest tests/
```

**No GNU Make on your machine?** (plain Windows without MinGW/choco/scoop)
Run the same three steps directly:

```bash
python scripts/run_ingest.py
python scripts/run_backtest.py
streamlit run app/streamlit_app.py
```

By default, `run_ingest.py`/`run_backtest.py` cover the **most recent 60
days**, not the full historical range — fetching years of data against
three live external APIs on a first `make ingest` would be a multi-hour,
easy-to-fail operation, not something that should happen silently. Pass
`--start`/`--end` (YYYY-MM-DD) for a full backfill, e.g.:

```bash
python scripts/run_ingest.py --start 2024-01-01 --end 2024-12-31
python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31 --markets day_ahead --models lgbm_quantile
```

## Repo structure

```
config/markets.yaml    per-market gate closure / cutoff / product config — the
                        single source of truth every timing decision reads from
data/raw/               immutable cached downloads (gitignored)
data/processed/         intermediate feature artifacts (gitignored)
src/ingest/             ENTSO-E, regelleistung.net, Open-Meteo clients
src/features/           availability enforcement + feature builder
src/backtest/           cutoff generation + walk-forward engine
src/models/             model registry (ForecastModel implementations)
src/evaluation/         point + probabilistic metrics
app/                    Streamlit monitoring UI
results/                backtest output (parquet + JSON sidecar, gitignored)
scripts/                CLI entry points for make targets
tests/                  pytest suite (all synthetic/fast except where noted)
```

## Market cutoff logic

Every market's timing lives in `config/markets.yaml` and nowhere else —
`src/backtest/cutoffs.py` (cutoff generation) and `src/features/availability.py`
(feature availability) both read it; a test
(`tests/test_backtest_no_hardcoded_timestamps.py`) fails the build if a
market-specific clock time ever gets hard-coded into `src/backtest`
instead.

| Market | Gate closure (D-1) | Forecast cutoff (D-1) | Product |
|---|---|---|---|
| `day_ahead` | 12:00 | 11:00 | 1h (24 targets/day) |
| `fcr` | 08:00 | 07:00 | 4h block (6 targets/day) |
| `afrr_pos` / `afrr_neg` | 09:00 | 08:00 | 4h block (6 targets/day) |

The forecast cutoff is always set 1h before gate closure — the last
moment a forecast could realistically inform a bid.

**This determines which features are even legal to use**, not just when
the forecast runs. `src/features/availability.py` gives every feature an
`available_at(target_time)` rule and drops (loudly, logged) anything not
knowable by the cutoff. Two consequences that fell out of this, not put
in by hand:

- A `lag_1h` feature (previous hour's actual price) is **never** usable
  for `day_ahead`, because even the very first hour of the delivery day
  is still 13h ahead of the 11:00 D-1 cutoff — "yesterday's last hour"
  hasn't happened yet when the forecast is made.
- ENTSO-E's day-ahead load/wind/solar forecasts publish D-1 18:00
  (verified against the Transparency Platform's own regulation text) —
  *after* every market's cutoff here. They are ingested (`src/ingest/entsoe.py`)
  but deliberately not wired into the feature builder, since they'd be
  dropped on every single call.

FCR and aFRR build features directly at their native 4h block granularity
(never hourly-then-averaged) — a "1 period ago" lag means the previous 4h
block, not the previous hour.

## Adding a new model

1. Create `src/models/your_model.py` implementing the `ForecastModel`
   interface (`src/models/base.py`):

   ```python
   class YourModel:
       name = "your_model"
       supports_quantiles = False  # or True

       def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
       def predict(self, X: pd.DataFrame, quantiles=None) -> pd.DataFrame:
           ...  # returns a DataFrame indexed like X, one column per quantile
           # (point models: a single "0.5" column)
   ```

   `X` has already passed the availability filter — every column is
   legal to use for whatever cutoff it was built at. Exclude
   `target_time` (metadata, not a feature) via
   `src.models.base.select_feature_columns`.

2. Register it in `src/models/registry.py`'s `MODEL_REGISTRY` dict.

No changes to `src/backtest/engine.py` or `app/` are needed — both only
ever go through the registry.

## Testing

```bash
pytest tests/ -q
```

All 39 tests are fast/synthetic except where the milestone explicitly
called for checking real behavior (e.g. verifying ENTSO-E/regelleistung
endpoint formats against the live services during development) — those
checks aren't in the permanent suite, by design, so `pytest` never needs
network access or API keys to pass.

## Findings from the current backtest (day_ahead + fcr, Jan 2024)

- **Which model wins depends on the market**, not just on model
  complexity. On `day_ahead`, every model beats `seasonal_naive`, and
  `lgbm_quantile`/`lgbm_quantile_tuned` win clearly (skill score vs. naive
  ≈ 0.52–0.54). On `fcr`, the ranking partially inverts: `linear_lags`
  (skill −0.45) and `lgbm_point` (skill −0.03) are actually *worse* than
  the naive "same block last week" baseline. FCR capacity prices are much
  more autocorrelated/flat than day-ahead energy prices, and with only
  ~84 test rows a simple naive lookup is a genuinely hard baseline to
  beat — throwing a bigger model at a market doesn't automatically help.
- **The quantile intervals are miscalibrated**, and this only shows up if
  you check coverage rather than RMSE. `lgbm_quantile` on `day_ahead` has
  RMSE-based skill of ~0.54 (looks good), but its 50%/80%/90% intervals'
  *empirical* coverage is ~24%/47%/58% — well below nominal at every
  level. The model is meaningfully overconfident: its intervals are too
  narrow for how often reality actually falls outside them. This is
  exactly the failure mode Milestone 6 was built to catch, and it's
  visible immediately on the Calibration page's reliability diagram
  (points sit well below the diagonal) — a bidding decision based on
  this model's stated uncertainty would be riskier than it looks.

## Explicitly out of scope

No battery model, no dispatch optimisation, no revenue stacking, no
market-selection logic. The backtest output schema (`results/*.parquet`)
is deliberately long-format and indexed by delivery period + quantile
specifically so a future battery optimiser could consume it directly —
that layer isn't built here.
