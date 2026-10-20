"""LEAR (Lago, Marcjasz, De Schutter, Weron 2021): one LASSO regression per period of the
day, fed with the whole price profile of the last days, on an asinh-transformed price scale.
PLAN_V8_REVISION.md V8-4.

Inputs per delivery period: the `lear_p_*` price-history columns (hourly mean prices of 1, 2,
3 and 7 days ago for day-ahead, block prices for FCR/aFRR), the wind/solar forecast of the
delivery period, and weekday dummies. The LASSO strength is chosen by cross-validation inside
the training window only (never on the test window).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.features.build import LEAR_PREFIX

logger = logging.getLogger(__name__)

MIN_ROWS_PER_PERIOD = 20  # below this a period gets the plain median instead of a LASSO
CV_SPLITS = 4
# A standardised input is held within +-INPUT_CLIP standard deviations, and a prediction within the range
# of that period's training targets: an input that was almost constant in training (solar at 20:45 in
# winter) can otherwise explode once it moves (first spring evenings), and sinh() turns that into 1e198.
INPUT_CLIP = 5.0
_NON_PRICE_SUFFIX = "_forecast_mw"


class LEAR:
    name = "lear"
    supports_quantiles = False

    def __init__(self, cv_splits: int = CV_SPLITS, max_iter: int = 20000):
        self.cv_splits = cv_splits
        self.max_iter = max_iter
        self._periods: dict[tuple[int, int], tuple[StandardScaler, LassoCV, float, float] | float] = {}
        self._columns: list[str] | None = None
        self._price_columns: list[str] = []
        self._column_means: pd.Series | None = None
        self._median = self._mad = 0.0

    def _asinh(self, values):
        return np.arcsinh((values - self._median) / self._mad)

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transformed prices, wind/solar forecasts and weekday dummies, in the fitted column order."""
        design = X.reindex(columns=self._columns[: len(self._columns) - 7]).copy()
        design[self._price_columns] = self._asinh(design[self._price_columns])
        dummies = pd.DataFrame({f"weekday_{d}": (X["day_of_week"] == d).astype(float) for d in range(7)}, index=X.index)
        return pd.concat([design, dummies], axis=1)

    @staticmethod
    def _period_of(X: pd.DataFrame) -> list[tuple[int, int]]:
        times = pd.DatetimeIndex(X["target_time"])
        return list(zip(times.hour, times.minute))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        price_columns = [c for c in X.columns if c.startswith(LEAR_PREFIX)]
        if not price_columns:
            raise ValueError("lear needs the lear_p_* price-history columns (src.features.build.add_price_history_features)")
        y = y.loc[X.index].astype(float)
        self._median = float(y.median())
        self._mad = float((y - self._median).abs().median()) or 1.0
        generation = [c for c in X.columns if c.endswith(_NON_PRICE_SUFFIX)]
        self._price_columns = price_columns
        self._columns = [*price_columns, *generation, *[f"weekday_{d}" for d in range(7)]]

        design = self._design(X)
        keep = design[price_columns].notna().all(axis=1) & y.notna()
        design, z, periods = design[keep], self._asinh(y[keep]), np.array(self._period_of(X))[keep.to_numpy()]
        self._column_means = design.mean()
        design = design.fillna(self._column_means)

        self._periods = {}
        for period in sorted({tuple(p) for p in periods}):
            rows = (periods == period).all(axis=1)
            if rows.sum() < MIN_ROWS_PER_PERIOD:
                self._periods[period] = float(z[rows].median())
                continue
            scaler = StandardScaler().fit(design[rows])
            lasso = LassoCV(cv=TimeSeriesSplit(self.cv_splits), max_iter=self.max_iter, n_alphas=60)
            lasso.fit(scaler.transform(design[rows]), z[rows])
            self._periods[period] = (scaler, lasso, float(z[rows].min()), float(z[rows].max()))

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        if self._columns is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        design = self._design(X)
        n_missing = int(design.isna().any(axis=1).sum())
        if n_missing:
            # A missing price slot (e.g. the 02:00 hour on the spring clock change) is replaced by the
            # training mean of that input, so the LASSO can still score the row; the price itself is never filled.
            logger.warning("lear: %d row(s) with a missing input replaced by the training mean of that input", n_missing)
        design = design.fillna(self._column_means)
        z_pred = np.empty(len(X))
        fallback = float(np.median([m for m in self._periods.values() if isinstance(m, float)] or [0.0]))
        for i, period in enumerate(self._period_of(X)):
            model = self._periods.get(period, fallback)
            if isinstance(model, float):
                z_pred[i] = model
            else:
                scaler, lasso, z_low, z_high = model
                inputs = np.clip(scaler.transform(design.iloc[[i]]), -INPUT_CLIP, INPUT_CLIP)
                z_pred[i] = np.clip(lasso.predict(inputs)[0], z_low, z_high)
        return pd.DataFrame({"0.5": np.sinh(z_pred) * self._mad + self._median}, index=X.index)

    def diagnostics(self) -> dict:
        """How many inputs the LASSOs kept, for the run metadata."""
        fitted = [m for m in self._periods.values() if not isinstance(m, float)]
        nonzero = [int((m[1].coef_ != 0).sum()) for m in fitted]
        return {
            "n_period_models": len(self._periods),
            "n_inputs": len(self._columns),
            "mean_nonzero_weights": float(np.mean(nonzero)) if nonzero else 0.0,
            "max_nonzero_weights": max(nonzero, default=0),
        }
