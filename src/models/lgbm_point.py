"""Plain LightGBM point forecast. LightGBM handles NaN natively, so —
unlike linear_lags — features are passed through without fillna."""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from src.models.base import select_feature_columns


class LGBMPoint:
    name = "lgbm_point"
    supports_quantiles = False

    def __init__(self, **lgbm_params):
        self.lgbm_params = lgbm_params
        self.model = LGBMRegressor(**lgbm_params)
        self.feature_columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = select_feature_columns(X)
        self.feature_columns = list(Xf.columns)
        self.model.fit(Xf, y.loc[X.index])

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        if self.feature_columns is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        Xf = X.reindex(columns=self.feature_columns)
        preds = self.model.predict(Xf)
        return pd.DataFrame({"0.5": preds}, index=X.index)
