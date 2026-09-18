"""Optuna-tuned LightGBM quantile regression. Tunes:
- which of the already-built lag_* columns to use (the "lags" dimension —
  we select among Milestone 3's availability-safe lag features rather
  than inventing new lag windows, which would require reaching back into
  the feature builder / raw ingestion from inside a model),
- num_leaves, learning_rate, n_estimators.

Optimises MEAN PINBALL LOSS across all quantiles, not RMSE: RMSE only
scores the median and says nothing about whether the other quantiles are
calibrated, which is the entire point of a quantile model.
"""
from __future__ import annotations

import pandas as pd
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

from src.models.base import select_feature_columns
from src.models.lgbm_quantile import DEFAULT_QUANTILES

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMQuantileTuned:
    name = "lgbm_quantile_tuned"
    supports_quantiles = True

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_trials: int = 30,
        random_state: int = 0,
    ):
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.n_trials = n_trials
        self.random_state = random_state
        self.models: dict[float, LGBMRegressor] = {}
        self.feature_columns: list[str] | None = None
        self.selected_columns: list[str] | None = None
        self.best_params: dict | None = None

    def _mean_pinball(self, y_true, y_pred_by_quantile: dict[float, pd.Series]) -> float:
        losses = [mean_pinball_loss(y_true, y_pred_by_quantile[q], alpha=q) for q in self.quantiles]
        return sum(losses) / len(losses)

    def _objective(self, trial: optuna.Trial, Xf: pd.DataFrame, yf: pd.Series) -> float:
        lag_cols = [c for c in Xf.columns if c.startswith("lag_")]
        other_cols = [c for c in Xf.columns if c not in lag_cols]
        chosen_lags = [c for c in lag_cols if trial.suggest_categorical(f"use_{c}", [True, False])]
        columns = other_cols + chosen_lags
        if not columns:
            return float("inf")  # no features at all — invalid trial

        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        }

        X_train, X_val, y_train, y_val = train_test_split(
            Xf[columns], yf, test_size=0.2, shuffle=False,  # time series: no shuffling
        )
        preds_by_q = {}
        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **params)
            model.fit(X_train, y_train)
            preds_by_q[q] = model.predict(X_val)
        return self._mean_pinball(y_val, preds_by_q)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xf = select_feature_columns(X)
        self.feature_columns = list(Xf.columns)
        yf = y.loc[X.index]

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(lambda trial: self._objective(trial, Xf, yf), n_trials=self.n_trials)

        best = study.best_params
        lag_cols = [c for c in Xf.columns if c.startswith("lag_")]
        other_cols = [c for c in Xf.columns if c not in lag_cols]
        self.selected_columns = other_cols + [c for c in lag_cols if best.get(f"use_{c}", False)]
        self.best_params = {
            k: v for k, v in best.items() if not k.startswith("use_")
        }

        for q in self.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **self.best_params)
            model.fit(Xf[self.selected_columns], yf)
            self.models[q] = model

    def predict(self, X: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
        if self.selected_columns is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        Xf = X.reindex(columns=self.selected_columns)
        requested = quantiles or self.quantiles
        out = {}
        for q in requested:
            out[str(q)] = self.models[q].predict(Xf)
        return pd.DataFrame(out, index=X.index)
