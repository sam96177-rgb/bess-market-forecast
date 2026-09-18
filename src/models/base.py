"""Common interface every forecast model implements, so the backtest
engine and UI can treat them interchangeably. A point forecast is
represented as a single "0.5"-labelled column, matching Milestone 5's
output schema convention (quantile = 0.5 for point forecasts, the
quantile level otherwise) — callers never need to special-case point vs.
quantile models.

This deviates from the illustrative Protocol in the project brief
(`fit(train, covariates)` / `predict(cutoff, horizon, quantiles)`): cutoff,
horizon and covariate-window logic already live in src.backtest.cutoffs
and src.features.build. By the time a model sees data, it's just X
(already availability-filtered features) and y (the target) — a plain
scikit-learn-style `fit(X, y)` / `predict(X)` contract. The backtest
engine's job becomes: build features for a cutoff, hand them to whichever
registered model, done — no model needs to know what a "cutoff" is.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ForecastModel(Protocol):
    name: str
    supports_quantiles: bool

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        """Returns a DataFrame indexed like X, one column per requested
        quantile (keyed by its string level, e.g. "0.5"). Models with
        supports_quantiles=False ignore `quantiles` and always return a
        single "0.5" column."""
        ...


def select_feature_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Every model excludes target_time — it's metadata for joining back
    to results, never a predictor."""
    return X[[c for c in X.columns if c != "target_time"]]
