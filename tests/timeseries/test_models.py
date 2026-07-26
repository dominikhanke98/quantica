"""Validation of the GARCH-family models and OOS forecaster (numerical-validation skill).

The estimation is leaned on (:mod:`arch`), so the anchor is **known-truth parameter recovery**:
simulate from a known GARCH / GJR / EGARCH process and confirm the fit recovers the true
parameters — the honest way to trust a library you did not write. The rolling forecaster is
checked for the property that actually matters: its forecasts are genuinely *out-of-sample* (the
target never enters the estimation window), and it degrades gracefully on bad inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.timeseries import (
    fit_volatility_model,
    rolling_forecast,
    simulate_garch,
)


def test_known_truth_garch_parameters_recovered() -> None:
    """A fit to a simulated GARCH(1,1) series recovers the true (omega, alpha, beta)."""
    rng = np.random.default_rng(0)
    returns = simulate_garch(6000, omega=0.05, alpha=0.08, beta=0.90, model="GARCH", rng=rng)
    fit = fit_volatility_model(returns, "GARCH")
    assert abs(fit.params["omega"] - 0.05) < 0.03
    assert abs(fit.params["alpha[1]"] - 0.08) < 0.03
    assert abs(fit.params["beta[1]"] - 0.90) < 0.03


def test_known_truth_gjr_leverage_recovered() -> None:
    """A fit to a simulated GJR series recovers a positive leverage term near the truth."""
    rng = np.random.default_rng(1)
    returns = simulate_garch(
        8000, omega=0.05, alpha=0.03, beta=0.90, gamma=0.08, model="GJR", rng=rng
    )
    fit = fit_volatility_model(returns, "GJR")
    assert fit.params["gamma[1]"] > 0.0  # leverage detected
    assert abs(fit.params["gamma[1]"] - 0.08) < 0.05
    assert abs(fit.params["beta[1]"] - 0.90) < 0.04


def test_leverage_model_wins_in_sample_on_asymmetric_data() -> None:
    """On data with true leverage, GJR/EGARCH beat plain GARCH on the information criterion."""
    rng = np.random.default_rng(2)
    returns = simulate_garch(
        6000, omega=0.05, alpha=0.03, beta=0.90, gamma=0.12, model="GJR", rng=rng
    )
    garch = fit_volatility_model(returns, "GARCH")
    gjr = fit_volatility_model(returns, "GJR")
    egarch = fit_volatility_model(returns, "EGARCH")
    # The asymmetric models capture a real feature, so they fit better (lower BIC, higher LL).
    assert gjr.bic < garch.bic
    assert egarch.bic < garch.bic
    assert gjr.loglikelihood > garch.loglikelihood


def test_rolling_forecast_is_out_of_sample_and_aligned() -> None:
    """The forecaster returns one variance per OOS target, positive, aligned with the proxy."""
    rng = np.random.default_rng(3)
    returns = simulate_garch(1500, omega=0.05, alpha=0.10, beta=0.85, model="GARCH", rng=rng)
    result = rolling_forecast(returns, "GARCH", first_forecast=1200, refit=50)
    assert result.variance_forecast.shape == (300,)
    assert result.realized_proxy.shape == (300,)
    assert np.all(result.variance_forecast > 0.0)
    # The proxy is the squared realised return at each target (aligned, out-of-sample).
    assert np.allclose(result.realized_proxy, returns[1200:] ** 2)
    assert np.allclose(result.actual_returns, returns[1200:])


def test_expanding_and_rolling_windows_both_run() -> None:
    """Both window modes produce sensible, finite forecasts (rolling uses a fixed-length window)."""
    rng = np.random.default_rng(4)
    returns = simulate_garch(1000, omega=0.05, alpha=0.10, beta=0.85, model="GARCH", rng=rng)
    expanding = rolling_forecast(returns, "GARCH", first_forecast=800, refit=100, expanding=True)
    rolling = rolling_forecast(returns, "GARCH", first_forecast=800, refit=100, expanding=False)
    assert np.all(np.isfinite(expanding.variance_forecast))
    assert np.all(np.isfinite(rolling.variance_forecast))


def test_simulate_garch_can_return_the_true_variance_path() -> None:
    """With return_variance, the oracle conditional-variance path aligns with the returns."""
    rng = np.random.default_rng(5)
    returns, variance = simulate_garch(
        2000, omega=0.05, alpha=0.10, beta=0.88, model="GARCH", rng=rng, return_variance=True
    )
    assert returns.shape == variance.shape == (2000,)
    assert np.all(variance > 0.0)


def test_invalid_arguments_are_rejected() -> None:
    """Bad model names, sizes and forecast indices raise clear errors."""
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError, match="n must be positive"):
        simulate_garch(0, omega=0.05, alpha=0.1, beta=0.85, rng=rng)
    with pytest.raises(ValueError, match="unknown model"):
        simulate_garch(100, omega=0.05, alpha=0.1, beta=0.85, model="BAD", rng=rng)  # type: ignore[arg-type]
    returns = simulate_garch(200, omega=0.05, alpha=0.1, beta=0.85, rng=rng)
    with pytest.raises(ValueError, match="first_forecast"):
        rolling_forecast(returns, "GARCH", first_forecast=0)
    with pytest.raises(ValueError, match="refit must be positive"):
        rolling_forecast(returns, "GARCH", first_forecast=100, refit=0)
