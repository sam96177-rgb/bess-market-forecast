"""Model registry: adding a new model means adding one file (a class
implementing src.models.base.ForecastModel) and one entry below — no
changes to the backtest engine or UI.
"""
from __future__ import annotations

from src.models.base import ForecastModel
from src.models.lgbm_point import LGBMPoint
from src.models.lgbm_quantile import LGBMQuantile
from src.models.lgbm_quantile_tuned import LGBMQuantileTuned
from src.models.linear_lags import LinearLags
from src.models.seasonal_naive import SeasonalNaive

MODEL_REGISTRY: dict[str, type] = {
    "seasonal_naive": SeasonalNaive,
    "linear_lags": LinearLags,
    "lgbm_point": LGBMPoint,
    "lgbm_quantile": LGBMQuantile,
    "lgbm_quantile_tuned": LGBMQuantileTuned,
}


def get_model(name: str, **kwargs) -> ForecastModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
