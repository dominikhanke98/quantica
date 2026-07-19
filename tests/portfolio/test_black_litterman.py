"""Validation of Black--Litterman (numerical-validation skill).

Known-truth anchors: reverse optimisation round-trips (equilibrium returns fed back
into mean-variance recover the market weights to machine precision), and with **no
views** the posterior collapses to the equilibrium prior. A view pushes the posterior in
its direction. The **headline** is the stability contrast that is the reason BL exists:
perturbing the return inputs swings naive mean-variance weights wildly, while BL —
shrinking toward equilibrium — barely moves.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.portfolio.black_litterman import (
    black_litterman,
    implied_equilibrium_returns,
)
from quantica.portfolio.construction import mean_variance_weights


def _well_conditioned_cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((400, n)) @ rng.standard_normal((n, n)) * 0.05
    return np.cov(returns, rowvar=False)


# --------------------------------------------------------------------------- #
# Known-truth anchors
# --------------------------------------------------------------------------- #


def test_reverse_optimization_round_trips() -> None:
    """π = δΣw_mkt fed back into mean-variance recovers w_mkt to machine precision."""
    cov = _well_conditioned_cov(6, seed=1)
    w_mkt = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
    delta = 2.5
    pi = implied_equilibrium_returns(cov, w_mkt, delta)
    recovered = mean_variance_weights(pi, cov, delta)  # budget-constrained MV
    assert np.allclose(recovered, w_mkt, atol=1e-8)


def test_no_views_posterior_is_the_equilibrium() -> None:
    cov = _well_conditioned_cov(5, seed=2)
    w_mkt = np.full(5, 0.2)
    delta = 3.0
    result = black_litterman(cov, w_mkt, delta)
    pi = implied_equilibrium_returns(cov, w_mkt, delta)
    assert np.allclose(result.posterior_returns, pi, atol=1e-12)
    assert np.allclose(result.equilibrium_returns, pi, atol=1e-12)
    assert np.allclose(result.posterior_cov, cov, atol=1e-12)


def test_a_view_moves_the_posterior_toward_it() -> None:
    # A confident bullish absolute view on asset 0 lifts its posterior return above the
    # equilibrium; a low-confidence view barely moves it.
    cov = _well_conditioned_cov(4, seed=3)
    w_mkt = np.full(4, 0.25)
    delta = 2.5
    pi = implied_equilibrium_returns(cov, w_mkt, delta)

    p = np.array([[1.0, 0.0, 0.0, 0.0]])  # a view on asset 0 alone
    q = np.array([pi[0] + 0.05])  # bullish relative to equilibrium
    tight = np.array([[1e-8]])  # high confidence
    loose = np.array([[1.0]])  # low confidence

    posterior_tight = black_litterman(
        cov, w_mkt, delta, views_p=p, views_q=q, view_uncertainty=tight
    )
    posterior_loose = black_litterman(
        cov, w_mkt, delta, views_p=p, views_q=q, view_uncertainty=loose
    )

    assert posterior_tight.posterior_returns[0] > pi[0]  # moved toward the bullish view
    # The confident view moves the posterior more than the diffuse one.
    assert (
        posterior_tight.posterior_returns[0] - pi[0] > posterior_loose.posterior_returns[0] - pi[0]
    )


# --------------------------------------------------------------------------- #
# The headline: BL stabilises mean-variance under input perturbation
# --------------------------------------------------------------------------- #


def test_black_litterman_stabilises_mean_variance() -> None:
    """Perturbing the return inputs swings naive MV weights far more than BL weights.

    Naive mean-variance treats noisy return estimates as truth and inverts the
    covariance, so small input changes produce large weight swings. Black--Litterman
    expresses the same estimates as views blended toward the equilibrium, so the
    posterior — and the weights — barely move. Measured here as the mean L1 weight change
    under 1% return perturbations (BL swings an order of magnitude less).
    """
    cov = _well_conditioned_cov(6, seed=1)
    w_mkt = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
    delta = 2.5
    n = 6
    base = implied_equilibrium_returns(cov, w_mkt, delta)  # a return estimate near equilibrium

    def naive_mv_weights(mu: np.ndarray) -> np.ndarray:
        return mean_variance_weights(mu, cov, delta)

    def bl_weights(view_shift: np.ndarray) -> np.ndarray:
        result = black_litterman(cov, w_mkt, delta, views_p=np.eye(n), views_q=base + view_shift)
        return mean_variance_weights(result.posterior_returns, result.posterior_cov, delta)

    rng = np.random.default_rng(7)
    mv_swings, bl_swings = [], []
    for _ in range(25):
        eps = rng.normal(0.0, 0.01, n)  # 1% perturbation of the return estimates
        mv_swings.append(
            float(np.sum(np.abs(naive_mv_weights(base + eps) - naive_mv_weights(base))))
        )
        bl_swings.append(float(np.sum(np.abs(bl_weights(eps) - bl_weights(np.zeros(n))))))

    # BL weights are far more stable than naive mean-variance (measured ~20x).
    assert np.mean(bl_swings) < np.mean(mv_swings) / 5.0


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_invalid_inputs_raise() -> None:
    cov = _well_conditioned_cov(4, seed=5)
    w = np.full(4, 0.25)
    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        implied_equilibrium_returns(cov, w, 0.0)
    with pytest.raises(ValueError, match="tau must be positive"):
        black_litterman(cov, w, 2.5, tau=0.0)
    with pytest.raises(ValueError, match="must have 4 columns"):
        black_litterman(cov, w, 2.5, views_p=np.eye(3), views_q=np.zeros(3))
    with pytest.raises(ValueError, match="rows"):
        black_litterman(cov, w, 2.5, views_p=np.eye(4), views_q=np.zeros(3))
