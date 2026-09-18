"""Benchmark every other model must beat: same-period-last-week. Directly
reuses the lag_168h feature Milestone 3's build_feature_frame already
produces (168h = 1 week for hourly markets; for FCR/aFRR's 4h blocks it's
still 168h = 1 week, just 42 blocks back). Stateless: fit() learns nothing.
"""
from __future__ import annotations

import pandas as pd

LAG_COLUMN = "lag_168h"


class SeasonalNaive:
    name = "seasonal_naive"
    supports_quantiles = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        if LAG_COLUMN not in X.columns:
            raise ValueError(
                f"{self.name} requires '{LAG_COLUMN}' in X. If it's missing, the "
                "availability filter dropped it for this cutoff — meaning even a "
                "1-week-ago lookup isn't valid here, so there is no legitimate "
                "seasonal_naive prediction to make."
            )
        return pd.DataFrame({"0.5": X[LAG_COLUMN].values}, index=X.index)
