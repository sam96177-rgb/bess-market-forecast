"""LightGBM quantile regression: one model per quantile level. LightGBM's
native `objective="quantile"` is single-quantile-per-model — there's no
built-in multi-quantile mode the way Darts' LightGBMModel(likelihood=
"quantile") wrapper provides. We deliberately don't use that Darts wrapper
here: see Milestone 4 checkpoint notes — it builds its own lagged feature
matrix internally from the raw target series, which would bypass Milestone
3's availability filter and risk reintroducing the leakage bug that filter
exists to catch. Training N independent models on our own already-safe X
is simpler and doesn't have that risk.
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from src.models.base import select_feature_columns

DEFAULT_QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


class LGBMQuantile:
    name = "lgbm_quantile"
    supports_quantiles = True

    def __init__(self, quantiles: list[float] | None = None, **lgbm_params):
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.lgbm_params = lgbm_params
        self.models: dict[float, LGBMRegressor] = {}
        self.feature_columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = select_feature_columns(X)
        self.feature_columns = list(Xf.columns)
        yf = y.loc[X.index]
        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **self.lgbm_params)
            model.fit(Xf, yf)
            self.models[q] = model

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        Xf = X.reindex(columns=self.feature_columns)
        requested = quantiles or self.quantiles
        out = {}
        for q in requested:
            if q not in self.models:
                raise ValueError(f"Quantile {q} was not trained; trained: {list(self.models)}")
            out[str(q)] = self.models[q].predict(Xf)
        return pd.DataFrame(out, index=X.index)
