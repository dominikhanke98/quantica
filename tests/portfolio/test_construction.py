"""Validation of the constrained portfolio constructors.

The headline anchors: the cvxpy minimum-variance and mean-variance portfolios reduce
**exactly** to their closed forms in the unconstrained (budget-only) case, and
risk-parity reduces to inverse-volatility weights for a diagonal covariance — so the
solver is validated against algebra, not merely against itself. The constraint tests
assert the long-only, position-limit and turnover budgets are actually respected.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.factor.estimators import min_variance_weights as closed_form_gmv
from quantica.portfolio.construction import (
    PortfolioConstraints,
    mean_variance_weights,
    minimum_variance_weights,
    risk_parity_weights,
)


def _sample_cov(n: int, t: int, seed: int) -> np.ndarray:
    """A well-conditioned sample covariance from a seeded return panel."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((t, n)) @ rng.standard_normal((n, n)) * 0.05
    return np.cov(returns, rowvar=False)


# --------------------------------------------------------------------------- #
# Minimum variance
# --------------------------------------------------------------------------- #


def test_minimum_variance_matches_closed_form() -> None:
    """Budget-only GMV equals w ∝ Σ⁻¹1 to solver precision (the anchor)."""
    cov = _sample_cov(6, 120, seed=1)
    w = minimum_variance_weights(cov)
    assert np.allclose(w, closed_form_gmv(cov), atol=1e-7)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_minimum_variance_is_the_minimiser() -> None:
    """The GMV variance is below that of nearby feasible portfolios."""
    cov = _sample_cov(5, 120, seed=2)
    w = minimum_variance_weights(cov)
    gmv_var = float(w @ cov @ w)
    rng = np.random.default_rng(0)
    for _ in range(50):
        perturb = rng.normal(0.0, 0.01, size=5)
        w_alt = w + perturb - perturb.mean()  # stay on the budget hyperplane
        assert w_alt @ cov @ w_alt >= gmv_var - 1e-12


def test_long_only_forbids_shorts() -> None:
    cov = _sample_cov(8, 30, seed=3)  # few obs → unconstrained GMV would short
    w_ls = minimum_variance_weights(cov)
    assert w_ls.min() < 0.0  # the long-short GMV does short here
    w_lo = minimum_variance_weights(cov, PortfolioConstraints(long_only=True))
    assert w_lo.min() >= -1e-9
    assert np.isclose(w_lo.sum(), 1.0, atol=1e-8)


def test_position_limit_is_respected() -> None:
    cov = _sample_cov(6, 120, seed=4)
    cap = 0.25
    w = minimum_variance_weights(cov, PortfolioConstraints(long_only=True, max_position=cap))
    assert w.max() <= cap + 1e-8
    assert np.isclose(w.sum(), 1.0, atol=1e-8)


def test_turnover_budget_is_respected() -> None:
    cov = _sample_cov(6, 120, seed=5)
    w_prev = np.full(6, 1.0 / 6.0)
    budget = 0.20
    w = minimum_variance_weights(cov, PortfolioConstraints(max_turnover=budget), w_prev=w_prev)
    assert np.sum(np.abs(w - w_prev)) <= budget + 1e-7
    # Without the budget the optimiser trades more than the cap allows.
    w_free = minimum_variance_weights(cov)
    assert np.sum(np.abs(w_free - w_prev)) > budget


# --------------------------------------------------------------------------- #
# Mean variance
# --------------------------------------------------------------------------- #


def _closed_form_mean_variance(mu: np.ndarray, cov: np.ndarray, gamma: float) -> np.ndarray:
    """Budget-constrained MV closed form: w = (1/gamma)Σ⁻¹(μ + λ1), λ from 1ᵀw = 1."""
    inv1 = np.linalg.solve(cov, np.ones(len(mu)))
    inv_mu = np.linalg.solve(cov, mu)
    lam = (gamma - np.ones(len(mu)) @ inv_mu) / (np.ones(len(mu)) @ inv1)
    return (inv_mu + lam * inv1) / gamma


def test_mean_variance_matches_closed_form() -> None:
    cov = _sample_cov(5, 120, seed=6)
    rng = np.random.default_rng(7)
    mu = rng.normal(0.0, 0.01, size=5)
    gamma = 4.0
    w = mean_variance_weights(mu, cov, gamma)
    assert np.allclose(w, _closed_form_mean_variance(mu, cov, gamma), atol=1e-7)


def test_mean_variance_approaches_min_variance_as_risk_aversion_grows() -> None:
    cov = _sample_cov(5, 120, seed=8)
    rng = np.random.default_rng(9)
    mu = rng.normal(0.0, 0.01, size=5)
    gmv = minimum_variance_weights(cov)
    w = mean_variance_weights(mu, cov, risk_aversion=1e6)
    assert np.allclose(w, gmv, atol=1e-3)  # conic-solver precision at large gamma


def test_mean_variance_tilts_toward_high_expected_return() -> None:
    """A higher alpha on one asset raises its weight, holding covariance fixed."""
    cov = _sample_cov(4, 120, seed=10)
    base = np.zeros(4)
    tilt = base.copy()
    tilt[0] = 0.02
    w_base = mean_variance_weights(base, cov, risk_aversion=2.0)
    w_tilt = mean_variance_weights(tilt, cov, risk_aversion=2.0)
    assert w_tilt[0] > w_base[0]


def test_mean_variance_rejects_nonpositive_risk_aversion() -> None:
    cov = _sample_cov(3, 60, seed=11)
    with pytest.raises(ValueError, match="risk_aversion"):
        mean_variance_weights(np.zeros(3), cov, risk_aversion=0.0)


# --------------------------------------------------------------------------- #
# Risk parity
# --------------------------------------------------------------------------- #


def test_risk_parity_equalises_risk_contributions() -> None:
    cov = _sample_cov(7, 120, seed=12)
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)  # marginal-risk contributions
    rc /= rc.sum()
    assert np.allclose(rc, 1.0 / 7.0, atol=1e-4)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    assert w.min() > 0.0


def test_risk_parity_diagonal_is_inverse_volatility() -> None:
    """For a diagonal Σ the ERC weights are w_i proportional to 1/sigma_i (a closed-form anchor)."""
    sigmas = np.array([0.10, 0.20, 0.30, 0.40])
    cov = np.diag(sigmas**2)
    w = risk_parity_weights(cov)
    expected = (1.0 / sigmas) / np.sum(1.0 / sigmas)
    assert np.allclose(w, expected, atol=1e-4)  # conic-solver precision


def test_constructors_reject_non_square_cov() -> None:
    bad = np.ones((3, 4))
    with pytest.raises(ValueError, match="square"):
        minimum_variance_weights(bad)
    with pytest.raises(ValueError, match="square"):
        risk_parity_weights(bad)
