"""Validation of the fEGarch QMLE engine (numerical-validation skill).

Checkable without fEGarch fixtures: on a simple known model (GARCH(1,1)) with normal innovations the
estimator recovers the planted parameters from simulated data; the conditional log-likelihood equals
a hand computation on a tiny fixed input; and the Hessian-based standard errors are finite and
positive. The "matches fEGarch to tolerance" check is a deferred skipped stub (CLAUDE.md §12) until
the R output fixtures are committed.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    get_distribution,
    initial_variance,
    quasi_max_likelihood,
)

_VAR_NAMES = ("omega", "alpha", "beta")
_VAR_BOUNDS = ((1e-8, 5.0), (1e-8, 0.999), (1e-8, 0.999))


def _garch11(params: np.ndarray, y: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """A plain GARCH(1,1) recursion, pre-sample variance conditioned on the sample variance."""
    omega, alpha, beta = params
    n = y.size
    sigma2 = np.empty(n, dtype=np.float64)
    sigma2[0] = initial_variance(y)
    for t in range(1, n):
        sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _simulate_garch11(n: int, omega: float, alpha: float, beta: float, seed: int) -> np.ndarray:  # type: ignore[type-arg]
    """Simulate a GARCH(1,1) series with standard-normal innovations."""
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return eps


def test_known_truth_garch_parameters_recovered() -> None:
    """QMLE recovers the planted GARCH(1,1) parameters under normal innovations."""
    y = _simulate_garch11(6000, omega=0.05, alpha=0.08, beta=0.90, seed=0)
    result = quasi_max_likelihood(
        y,
        _garch11,
        get_distribution("norm"),
        var_start=(0.1, 0.1, 0.8),
        var_bounds=_VAR_BOUNDS,
        var_names=_VAR_NAMES,
    )
    estimates = dict(zip(result.param_names, result.params, strict=True))
    assert result.converged
    assert abs(estimates["omega"] - 0.05) < 0.03
    assert abs(estimates["alpha"] - 0.08) < 0.03
    assert abs(estimates["beta"] - 0.90) < 0.03


def test_loglikelihood_matches_hand_computation() -> None:
    """The engine's log-likelihood equals a direct sum on a tiny fixed input."""
    y = np.array([0.1, -0.2, 0.15, -0.05, 0.3])
    params = np.array([0.05, 0.1, 0.85])
    dist = get_distribution("norm")

    sigma2 = _garch11(params, y)
    sigma = np.sqrt(sigma2)
    by_hand = float(np.sum(-np.log(sigma) + dist.logpdf(y / sigma)))

    # Fit with the true params pinned (degenerate bounds) so the reported loglik is at that point.
    result = quasi_max_likelihood(
        y,
        _garch11,
        dist,
        var_start=(0.05, 0.1, 0.85),
        var_bounds=((0.05, 0.05), (0.1, 0.1), (0.85, 0.85)),
        var_names=_VAR_NAMES,
    )
    assert np.isclose(result.loglikelihood, by_hand, atol=1e-9)
    assert np.allclose(result.conditional_variance, sigma2)


def test_standard_errors_are_finite_and_positive() -> None:
    """The Hessian-based standard errors behave sanely on a well-identified fit."""
    y = _simulate_garch11(6000, omega=0.05, alpha=0.08, beta=0.90, seed=1)
    result = quasi_max_likelihood(
        y,
        _garch11,
        get_distribution("norm"),
        var_start=(0.1, 0.1, 0.8),
        var_bounds=_VAR_BOUNDS,
        var_names=_VAR_NAMES,
    )
    assert result.std_errors.shape == (3,)
    assert np.all(np.isfinite(result.std_errors))
    assert np.all(result.std_errors > 0.0)
    assert np.all(result.std_errors < 0.1)  # tight-ish on 6000 obs


def test_distribution_shape_parameter_is_estimated() -> None:
    """Fitting with a Student-t innovation estimates its shape parameter within bounds."""
    y = _simulate_garch11(4000, omega=0.05, alpha=0.10, beta=0.85, seed=2)
    dist = get_distribution("std")
    result = quasi_max_likelihood(
        y,
        _garch11,
        dist,
        var_start=(0.1, 0.1, 0.8),
        var_bounds=_VAR_BOUNDS,
        var_names=_VAR_NAMES,
    )
    assert result.param_names[-1] == "nu"
    low, high = dist.param_bounds[0]
    assert low <= result.params[-1] <= high


def test_initial_variance_is_the_sample_variance() -> None:
    """The documented pre-sample conditioning returns the sample variance."""
    y = np.array([0.1, -0.3, 0.2, 0.05, -0.15])
    assert np.isclose(initial_variance(y), np.var(y))


def test_mismatched_variance_metadata_raises() -> None:
    """Inconsistent variance-parameter metadata lengths are rejected."""
    y = _simulate_garch11(200, 0.05, 0.08, 0.90, seed=3)
    with pytest.raises(ValueError, match="same length"):
        quasi_max_likelihood(
            y,
            _garch11,
            get_distribution("norm"),
            var_start=(0.1, 0.1, 0.8),
            var_bounds=((1e-8, 5.0), (1e-8, 0.999)),  # only two bounds
            var_names=_VAR_NAMES,
        )


# --------------------------------------------------------------------------- #
# Deferred: agreement with fEGarch's QMLE (needs committed R output fixtures)
# --------------------------------------------------------------------------- #


@pytest.mark.skip(
    reason="Phase 1: the fit fixtures (fit_garch11_norm_*, fit_egarch11_norm_*) are committed but "
    "matching them needs the fEGarch GARCH/EGARCH recursion + presample convention (Phase 1). "
    "This is the Phase-0 distributions + QMLE-engine PR."
)
def test_qmle_matches_fegarch_fixture() -> None:
    """The QMLE fit matches fEGarch's fitted parameters + conditional-variance series to tol."""
    raise AssertionError("fixture not available yet")  # pragma: no cover
