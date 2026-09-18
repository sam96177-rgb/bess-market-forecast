"""Ridge regression on whatever lag + calendar (+ weather, if it survived
the availability filter) features exist for this cutoff. Point forecast
only — Ridge has no native quantile mode."""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

from src.models.base import select_feature_columns


class LinearLags:
    name = "linear_lags"
    supports_quantiles = False

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.feature_columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = select_feature_columns(X).fillna(0.0)
        self.feature_columns = list(Xf.columns)
        self.model.fit(Xf, y.loc[X.index])

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        Xf = X.reindex(columns=self.feature_columns, fill_value=0.0).fillna(0.0)
        preds = self.model.predict(Xf)
        return pd.DataFrame({"0.5": preds}, index=X.index)
