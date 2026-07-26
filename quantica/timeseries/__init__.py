r"""Time series & econometrics — Pillar V, built forecast-evaluation-first.

Volatility forecasting is a solved *estimation* problem — :mod:`arch` fits GARCH-family models
well — so this pillar leans on it for the fit and puts the demonstrable work where a
model-validation team actually adds value: **deciding whether one forecast is genuinely better
than another**, correctly.

This foundational step ships GARCH-family volatility modelling and its forecast-evaluation layer:

* **Volatility models** (:mod:`~quantica.timeseries.models`) — ``GARCH(1,1)``, ``GJR-GARCH``
  (an asymmetry/leverage term) and ``EGARCH``, fitted via :mod:`arch`, with a rolling/expanding
  **out-of-sample** one-step-ahead variance forecaster.
* **Forecast evaluation** (:mod:`~quantica.timeseries.evaluation`) — the real deliverable: the
  **Diebold--Mariano** test of equal predictive accuracy with a **HAC / Newey--West** variance
  correction (the piece most implementations get wrong), the proxy-robust **QLIKE** and **MSE**
  loss functions (true volatility is latent, so forecasts are scored against a squared-return
  proxy), and the **Mincer--Zarnowitz** forecast-efficiency regression.
* **Synthetic data** (:mod:`~quantica.timeseries.data`) — known-parameter GARCH/GJR/EGARCH paths
  and serially-correlated loss differentials, the ground truth for validating the estimation and
  the evaluation statistics themselves.

The headline is *validate-the-validator*: on data with a known truth, the Diebold--Mariano test
has the correct size and power **only with** the HAC correction — the naive-variance version
over-rejects when loss differentials are serially correlated, which they are for volatility
forecasts.
"""

from __future__ import annotations

from quantica.timeseries.data import simulate_garch, simulate_loss_differential
from quantica.timeseries.evaluation import (
    DieboldMarianoResult,
    MincerZarnowitzResult,
    diebold_mariano,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
)
from quantica.timeseries.models import (
    ForecastResult,
    VolatilityFit,
    VolatilityModel,
    fit_volatility_model,
    rolling_forecast,
)

__all__ = [
    "DieboldMarianoResult",
    "ForecastResult",
    "MincerZarnowitzResult",
    "VolatilityFit",
    "VolatilityModel",
    "diebold_mariano",
    "fit_volatility_model",
    "mincer_zarnowitz",
    "mse_loss",
    "qlike_loss",
    "rolling_forecast",
    "simulate_garch",
    "simulate_loss_differential",
]
