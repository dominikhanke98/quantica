"""Validation of the VECM estimator (numerical-validation skill).

The headline is **known-truth recovery**: simulate a system from a planted VECM (known
cointegrating rank, cointegrating vectors and adjustment loadings) and confirm the Johansen
rank selection and the reduced-rank estimation recover them. Two anchors pin it down — the
coefficients match ``statsmodels``' ``VECM`` to near machine precision, and for a bivariate
rank-1 system the single cointegrating vector reduces to the pairwise hedge ratio the stat-arb
pillar estimates (the multivariate-generalises-pairwise claim, made concrete).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.statarb import engle_granger
from quantica.timeseries import fit_vecm, select_cointegration_rank, simulate_vecm

_ALPHA = np.array([[-0.10], [0.08], [0.0]])  # adjustment loadings (3 series, rank 1)
_BETA = np.array([[1.0], [-1.0], [0.0]])  # cointegrating vector: y0 - y1 stationary


def test_known_truth_rank_and_coefficients_recovered() -> None:
    """Johansen selects the true rank and the RRR recovers alpha, beta and Pi = alpha beta'."""
    y = simulate_vecm(4000, _ALPHA, _BETA, sigma=0.5, rng=np.random.default_rng(0))
    assert select_cointegration_rank(y, k_ar_diff=1) == 1

    fit = fit_vecm(y, rank=1, k_ar_diff=0, deterministic="n")
    # beta is Phillips-normalised (beta[0] = 1); recovers the true (1, -1, 0) direction.
    assert np.allclose(fit.beta[:, 0], [1.0, -1.0, 0.0], atol=0.05)
    assert np.allclose(fit.alpha[:, 0], _ALPHA[:, 0], atol=0.05)
    # The long-run impact matrix Pi = alpha beta' recovers the truth.
    assert np.allclose(fit.long_run_matrix, _ALPHA @ _BETA.T, atol=0.05)


def test_matches_statsmodels_vecm() -> None:
    """alpha and beta match statsmodels' VECM to near machine precision (same estimator)."""
    from statsmodels.tsa.vector_ar.vecm import VECM

    y = simulate_vecm(3000, _ALPHA, _BETA, sigma=0.5, rng=np.random.default_rng(1))
    fit = fit_vecm(y, rank=1, k_ar_diff=0, deterministic="n")
    reference = VECM(y, k_ar_diff=0, coint_rank=1, deterministic="n").fit()
    assert np.allclose(fit.beta, reference.beta, atol=1e-8)
    assert np.allclose(fit.alpha, reference.alpha, atol=1e-8)


def test_bivariate_reduces_to_pairwise_cointegration() -> None:
    """For n=2, rank 1, the VECM hedge ratio equals the pairwise Engle--Granger hedge ratio."""
    alpha = np.array([[-0.12], [0.05]])
    beta = np.array([[1.0], [-0.8]])  # true hedge ratio 0.8
    y = simulate_vecm(4000, alpha, beta, sigma=0.4, rng=np.random.default_rng(2))

    fit = fit_vecm(y, rank=1, k_ar_diff=0, deterministic="n")
    eg = engle_granger(y[:, 0], y[:, 1])
    assert abs(fit.hedge_ratio() - 0.8) < 0.05
    assert abs(fit.hedge_ratio() - eg.hedge_ratio) < 0.05


def test_short_run_dynamics_are_estimated() -> None:
    """With a lagged-difference term in the DGP, the fitted Gamma has the right shape and sign."""
    gamma = np.array([[0.3, 0.0], [0.0, 0.2]])
    y = simulate_vecm(
        5000,
        np.array([[-0.15], [0.10]]),
        np.array([[1.0], [-1.0]]),
        gamma=gamma,
        sigma=0.4,
        rng=np.random.default_rng(3),
    )
    fit = fit_vecm(y, rank=1, k_ar_diff=1, deterministic="co")
    assert fit.gamma.shape == (2, 2)
    assert np.allclose(np.diag(fit.gamma), np.diag(gamma), atol=0.08)


def test_invalid_arguments_are_rejected() -> None:
    """Out-of-range rank and malformed simulation inputs raise clear errors."""
    y = simulate_vecm(500, _ALPHA, _BETA, sigma=0.5, rng=np.random.default_rng(4))
    with pytest.raises(ValueError, match="rank must be in"):
        fit_vecm(y, rank=3, k_ar_diff=1)  # rank must be < n = 3
    with pytest.raises(ValueError, match="hedge_ratio is only defined"):
        fit_vecm(y, rank=1, k_ar_diff=0, deterministic="n").hedge_ratio()  # n=3, not bivariate
    with pytest.raises(ValueError, match="alpha and beta must have the same shape"):
        simulate_vecm(100, _ALPHA, _BETA[:2], rng=np.random.default_rng(5))
