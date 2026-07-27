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
* **Regime-switching** (:mod:`~quantica.timeseries.regime`) — a Gaussian Markov-switching model
  (calm vs crisis) with the hand-implemented **Hamilton filter**, **Kim smoother** and **EM**
  estimation, recovering the hidden volatility regimes a market moves between.
* **Multivariate** — the **VECM** (:mod:`~quantica.timeseries.vecm`), the multivariate
  generalisation of pairwise cointegration (Johansen reduced-rank estimation), and **DCC-GARCH**
  (:mod:`~quantica.timeseries.dcc`), a time-varying conditional covariance that feeds straight into
  the portfolio and risk pillars as a drop-in covariance estimator.
* **Synthetic data** (:mod:`~quantica.timeseries.data`) — known-parameter GARCH/GJR/EGARCH paths,
  serially-correlated loss differentials, Markov-switching series with their hidden states, and
  known VECM/DCC systems, the ground truth for validating the estimation and the statistics.

The headline is *validate-the-validator*: on data with a known truth, the Diebold--Mariano test
has the correct size and power **only with** the HAC correction — the naive-variance version
over-rejects when loss differentials are serially correlated, which they are for volatility
forecasts.
"""

from __future__ import annotations

from quantica.timeseries.data import (
    simulate_dcc,
    simulate_garch,
    simulate_loss_differential,
    simulate_markov_switching,
    simulate_vecm,
)
from quantica.timeseries.dcc import DccCovariance, DccResult, fit_dcc
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
from quantica.timeseries.regime import (
    HamiltonFilterResult,
    MarkovSwitchingResult,
    fit_markov_switching,
    hamilton_filter,
    kim_smoother,
)
from quantica.timeseries.vecm import VecmResult, fit_vecm, select_cointegration_rank

__all__ = [
    "DccCovariance",
    "DccResult",
    "DieboldMarianoResult",
    "ForecastResult",
    "HamiltonFilterResult",
    "MarkovSwitchingResult",
    "MincerZarnowitzResult",
    "VecmResult",
    "VolatilityFit",
    "VolatilityModel",
    "diebold_mariano",
    "fit_dcc",
    "fit_markov_switching",
    "fit_vecm",
    "fit_volatility_model",
    "hamilton_filter",
    "kim_smoother",
    "mincer_zarnowitz",
    "mse_loss",
    "qlike_loss",
    "rolling_forecast",
    "select_cointegration_rank",
    "simulate_dcc",
    "simulate_garch",
    "simulate_loss_differential",
    "simulate_markov_switching",
    "simulate_vecm",
]
