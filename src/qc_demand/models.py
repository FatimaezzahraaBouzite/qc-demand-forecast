"""Three forecasters behind one interface.

fit(train)                 -> train: feature frame (index = date), rows with known target
predict(frame, target_idx) -> one-step-ahead predictions for dates in target_idx.
                              `frame` holds all history up to and including those dates,
                              so each prediction only uses demand observed before its day.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .features import TARGET, feature_columns


class SeasonalNaive:
    """Same weekday last week. The baseline every other model must beat."""
    name = "seasonal_naive"

    def __init__(self, params: dict, **_):
        self.params = params

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        return self

    def predict(self, frame: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.Series:
        return frame.loc[target_idx, "lag_7"].rename("pred")


class SarimaxModel:
    """Seasonal ARIMA with weather/calendar regressors: the statistical benchmark."""
    name = "sarimax"
    EXOG = ["hdd", "cdd", "is_holiday", "is_xmas_break"]

    def __init__(self, params: dict, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)):
        self.params = params
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.res = None
        self.train_end = None

    def _scale(self, y: pd.Series, X: pd.DataFrame):
        return (y - self.mu) / self.sd, (X - self.x_mu) / self.x_sd

    def fit(self, train: pd.DataFrame) -> SarimaxModel:
        # Standardize: with demand ~20,000 MW the optimizer otherwise lands on
        # degenerate fits (AR and MA roots cancelling out).
        train = train.asfreq("D")
        y = train[TARGET].interpolate(limit_direction="both")
        X = train[self.EXOG].ffill().bfill().astype(float)
        self.mu, self.sd = y.mean(), y.std()
        self.x_mu, self.x_sd = X.mean(), X.std().replace(0, 1)
        ys, Xs = self._scale(y, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(ys, exog=Xs, order=self.order, seasonal_order=self.seasonal_order,
                          trend="c", enforce_stationarity=True, enforce_invertibility=True)
            self.res = mod.fit(disp=False, maxiter=300)
        self.train_end = train.index.max()
        return self

    def predict(self, frame: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.Series:
        hist = frame.loc[: target_idx.max()].asfreq("D")
        # one-step-ahead filtering: the prediction for day t only uses y up to t-1,
        # so passing actual (or NaN, in production) values for target days is leak-free
        ys, Xs = self._scale(hist[TARGET], hist[self.EXOG].ffill().bfill().astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self.res.apply(ys, exog=Xs, refit=False)
            pred = res.get_prediction(dynamic=False).predicted_mean * self.sd + self.mu
        return pred.loc[target_idx].rename("pred")


class LightGBMModel:
    """Gradient boosting on lag, calendar and weather features."""
    name = "lightgbm"

    def __init__(self, params: dict, **hyper):
        self.params = params
        self.hyper = hyper
        self.cols = feature_columns(params)
        self.model = None

    def fit(self, train: pd.DataFrame) -> LightGBMModel:
        self.model = lgb.LGBMRegressor(**self.hyper)
        self.model.fit(train[self.cols], train[TARGET])
        return self

    def predict(self, frame: pd.DataFrame, target_idx: pd.DatetimeIndex) -> pd.Series:
        X = frame.loc[target_idx, self.cols]
        return pd.Series(self.model.predict(X), index=target_idx, name="pred")

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.booster_.feature_importance("gain"),
                         index=self.cols).sort_values(ascending=False)


REGISTRY = {m.name: m for m in (SeasonalNaive, SarimaxModel, LightGBMModel)}


def make_model(name: str, params: dict):
    hyper = params["models"].get(name, {}) or {}
    return REGISTRY[name](params, **hyper)


# --------------------------------------------------------------------------- metrics
def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = y_true.align(y_pred, join="inner")
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask].to_numpy(), y_pred[mask].to_numpy()
    err = yp - yt
    return {
        "mape": float(np.mean(np.abs(err) / np.abs(yt)) * 100),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "n": int(mask.sum()),
    }
