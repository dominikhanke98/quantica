"""Validation of the DCC-GARCH model and its cross-pillar tie-back (numerical-validation skill).

The headline is **known-truth recovery of the time-varying correlation**: simulate a DCC process
with a known correlation path and confirm the fit recovers the parameters and tracks the path.
Two required properties are asserted directly — the conditional covariance is **positive-definite
at every step**, and the model **reduces to CCC** (constant conditional correlation) when the
dynamics vanish. The coherence payoff is the tie-back: :class:`DccCovariance` plugs into the factor
pillar's out-of-sample comparison harness, so DCC's dynamic covariance can be scored against the
static estimators on the very task those pillars care about.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.timeseries import DccCovariance, fit_dcc, simulate_dcc

# The DCC fixtures are deliberately unit-variance (to isolate the correlation dynamics), so arch's
# percent-scaling inside fit_dcc over-scales them; the resulting DataScaleWarning is cosmetic
# (standardised residuals are scale-invariant, so the correlation estimation is unaffected).
pytestmark = pytest.mark.filterwarnings("ignore:y is poorly scaled")

_QBAR = np.array([[1.0, 0.3], [0.3, 1.0]])


def test_known_truth_parameters_and_correlation_path_recovered() -> None:
    """The fit recovers the DCC (a, b) and tracks the true time-varying correlation."""
    returns, true_corr = simulate_dcc(3000, 0.04, 0.93, _QBAR, rng=np.random.default_rng(0))
    dcc = fit_dcc(returns)
    # The two-stage GARCH pre-filter makes the a/b split noisy in finite samples, but the news
    # impact, the high persistence and the total a + b are all recovered in a sensible band.
    assert abs(dcc.a - 0.04) < 0.04
    assert dcc.b > 0.80
    assert abs((dcc.a + dcc.b) - 0.97) < 0.12
    estimated_corr = dcc.conditional_correlations[:, 0, 1]
    # The estimated correlation path co-moves strongly with the true one (the real headline).
    assert np.corrcoef(estimated_corr, true_corr)[0, 1] > 0.7


def test_covariance_is_positive_definite_everywhere() -> None:
    """Every in-sample conditional covariance and the one-step forecast are positive-definite."""
    returns, _ = simulate_dcc(2000, 0.05, 0.90, _QBAR, rng=np.random.default_rng(1))
    dcc = fit_dcc(returns)
    for cov in dcc.conditional_covariances:
        assert np.linalg.eigvalsh(cov).min() > 0.0
    forecast = dcc.forecast_covariance()
    assert np.linalg.eigvalsh(forecast).min() > 0.0
    assert np.allclose(forecast, forecast.T)


def test_reduces_to_constant_correlation_when_dynamics_vanish() -> None:
    """On data with a truly constant correlation, DCC collapses to CCC (a + b ~ 0)."""
    returns, _ = simulate_dcc(3000, 0.0, 0.0, _QBAR, rng=np.random.default_rng(2))
    dcc = fit_dcc(returns)
    assert dcc.a + dcc.b < 0.15  # no dynamics detected
    # The conditional correlation is (near) constant across time — the CCC special case.
    assert dcc.conditional_correlations[:, 0, 1].std() < 0.05


def test_forecast_covariance_reflects_conditional_volatility() -> None:
    """The one-step forecast covariance combines the forecast vols and correlation coherently."""
    returns, _ = simulate_dcc(2000, 0.05, 0.90, _QBAR, rng=np.random.default_rng(3))
    dcc = fit_dcc(returns)
    forecast = dcc.forecast_covariance()
    implied_vol = np.sqrt(np.diag(forecast))
    assert np.allclose(implied_vol, dcc.forecast_volatility)
    implied_corr = forecast[0, 1] / (implied_vol[0] * implied_vol[1])
    assert np.isclose(implied_corr, dcc.forecast_correlation[0, 1])


def test_dcc_covariance_estimator_feeds_the_comparison_harness() -> None:
    """DccCovariance conforms to the factor CovarianceEstimator and runs in compare_estimators."""
    from quantica.factor import SampleCovariance, compare_estimators

    rng = np.random.default_rng(4)
    returns = np.column_stack(
        [simulate_dcc(400, 0.05, 0.9, _QBAR, rng=rng)[0][:, 0] for _ in range(3)]
    )
    estimator = DccCovariance()
    assert estimator.name == "dcc"
    cov = estimator.estimate(returns)
    assert cov.shape == (3, 3)
    assert np.linalg.eigvalsh(cov).min() > 0.0  # a valid (PD) covariance for the consuming pillars

    comparison = compare_estimators(
        returns,
        (SampleCovariance(), DccCovariance()),
        train_window=200,
        test_window=50,
        rng=np.random.default_rng(5),
        n_random_portfolios=10,
    )
    assert "dcc" in comparison.mean_min_variance_vol()


def test_invalid_arguments_are_rejected() -> None:
    """A univariate panel and a non-stationary correlation process raise clear errors."""
    with pytest.raises(ValueError, match="n >= 2"):
        fit_dcc(np.random.default_rng(6).normal(size=(100, 1)))
    with pytest.raises(ValueError, match="a \\+ b < 1"):
        simulate_dcc(100, 0.6, 0.6, _QBAR, rng=np.random.default_rng(7))
