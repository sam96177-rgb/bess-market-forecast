"""Fast, synthetic-data tests for the model registry: every registered
model must fit/predict on a plain X/y frame without errors, respect
supports_quantiles, and (for quantile models) produce quantile columns
that don't cross badly for a simple, learnable synthetic target — real
integration testing against actual market data happens in the Milestone 4
checkpoint demo, not here, so this suite stays fast.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.registry import MODEL_REGISTRY, get_model


def _synthetic_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    target_time = pd.date_range("2024-01-01", periods=n, freq="h", tz="Europe/Berlin")
    lag_168h = 50 + 10 * np.sin(np.arange(n) / 24) + rng.normal(0, 1, n)
    hour_of_day = target_time.hour
    X = pd.DataFrame({
        "target_time": target_time,
        "hour_of_day": hour_of_day,
        "lag_168h": lag_168h,
        "lag_672h": lag_168h + rng.normal(0, 0.5, n),
    })
    y = pd.Series(lag_168h + rng.normal(0, 0.5, n), index=X.index)
    return X, y


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_model_conforms_to_interface(model_name):
    X, y = _synthetic_frame()
    n_train = 150
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test = X.iloc[n_train:]

    kwargs = {"n_trials": 3} if model_name == "lgbm_quantile_tuned" else {}
    model = get_model(model_name, **kwargs)

    assert isinstance(model.name, str)
    assert isinstance(model.supports_quantiles, bool)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
    assert list(preds.index) == list(X_test.index)
    if model.supports_quantiles:
        assert set(preds.columns) >= {"0.05", "0.5", "0.95"}
    else:
        assert list(preds.columns) == ["0.5"]


def test_quantiles_are_reasonably_monotonic_for_lgbm_quantile():
    X, y = _synthetic_frame(n=400)
    X_train, y_train = X.iloc[:300], y.iloc[:300]
    X_test = X.iloc[300:]

    model = get_model("lgbm_quantile")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Quantile crossing can happen with independently-fit models, but on
    # average the median should sit between the 5th and 95th percentile.
    assert preds["0.05"].mean() <= preds["0.5"].mean() <= preds["0.95"].mean()


def test_seasonal_naive_is_exactly_the_lag_168h_column():
    X, y = _synthetic_frame(n=50)
    model = get_model("seasonal_naive")
    model.fit(X, y)  # no-op
    preds = model.predict(X)
    pd.testing.assert_series_equal(
        preds["0.5"], X["lag_168h"], check_names=False
    )


def test_seasonal_naive_raises_clearly_when_lag_168h_missing():
    X, y = _synthetic_frame(n=10)
    X_missing = X.drop(columns=["lag_168h"])
    model = get_model("seasonal_naive")
    model.fit(X_missing, y)
    with pytest.raises(ValueError, match="lag_168h"):
        model.predict(X_missing)


def test_lgbm_quantile_tuned_selects_a_subset_of_lag_columns():
    X, y = _synthetic_frame(n=300)
    model = get_model("lgbm_quantile_tuned", n_trials=5, random_state=1)
    model.fit(X.iloc[:250], y.iloc[:250])
    assert model.selected_columns is not None
    assert all(c in X.columns for c in model.selected_columns)


def test_unknown_model_name_raises_with_registered_list():
    with pytest.raises(KeyError, match="seasonal_naive"):
        get_model("not_a_real_model")
