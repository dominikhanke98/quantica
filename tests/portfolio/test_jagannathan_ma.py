r"""Cross-pillar insight: the no-short-sale constraint *is* covariance shrinkage.

Factor stage 2 found the sample covariance **worst** under (unconstrained) inversion —
its global minimum-variance portfolio realises ~2x the out-of-sample volatility of a
shrunk estimator (Michaud's "error maximiser"). Yet the portfolio backtest's best
net-of-cost strategy is `minvar/sample` — minimum-variance on the *same* sample
covariance, but **long-only**. These look contradictory; they are not.

Jagannathan & Ma (2003) proved the resolution: imposing a no-short-sale constraint on
the minimum-variance problem is *equivalent* to solving the unconstrained problem with
a **shrunk** covariance matrix

.. math::

    \tilde\Sigma = \Sigma - (\mu\mathbf 1^\top + \mathbf 1\mu^\top),

where :math:`\mu \ge 0` is the KKT multiplier on :math:`w \ge 0`. Assets that hit the
zero bound (:math:`\mu_i > 0`) have their covariances with every other asset reduced —
and those are exactly the high-covariance assets the unconstrained solution would
*short*. The constraint therefore regularises precisely the estimation error that
sinks the unconstrained sample-covariance portfolio.

These tests establish the equivalence to machine precision from the primal solution
alone (no dual solver needed): :math:`\lambda = w^\top\Sigma w` and
:math:`\mu = \Sigma w - \lambda\mathbf 1`, giving :math:`\tilde\Sigma w = \lambda\mathbf 1`
so that ``w`` is the unconstrained GMV of :math:`\tilde\Sigma`. A synthetic
out-of-sample test then confirms the practical consequence: long-only tames the
sample covariance.

References
----------
Jagannathan, R. & Ma, T. (2003), "Risk reduction in large portfolios: why imposing the
wrong constraints helps", *Journal of Finance* 58(4).
"""

from __future__ import annotations

import numpy as np
from quantica.factor.data import generate_factor_data
from quantica.factor.estimators import (
    LedoitWolfCovariance,
    SampleCovariance,
    min_variance_weights,
)
from quantica.portfolio.construction import PortfolioConstraints, minimum_variance_weights


def _ill_conditioned_sample_cov(n_assets: int, n_obs: int, seed: int) -> np.ndarray:
    """A sample covariance whose unconstrained GMV shorts (few obs, many assets)."""
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((n_obs, n_assets)) @ rng.standard_normal((n_assets, n_assets))
    return np.cov(returns * 0.05, rowvar=False)


def _implied_shrinkage(cov: np.ndarray, w_long_only: np.ndarray) -> np.ndarray:
    r"""The Jagannathan--Ma KKT multiplier mu from the long-only primal solution.

    From stationarity :math:`\Sigma w = \lambda\mathbf 1 + \mu` with
    :math:`\lambda = w^\top\Sigma w` (since :math:`\mu^\top w = 0`).
    """
    lam = float(w_long_only @ cov @ w_long_only)
    return np.asarray(cov @ w_long_only - lam, dtype=np.float64)


# --------------------------------------------------------------------------- #
# The exact equivalence (the mechanism)
# --------------------------------------------------------------------------- #


def test_long_only_gmv_equals_unconstrained_gmv_of_shrunk_cov() -> None:
    """Long-only GMV(Σ) == unconstrained GMV(Σ̃) to machine precision (Jagannathan-Ma)."""
    cov = _ill_conditioned_sample_cov(10, 40, seed=3)
    assert min_variance_weights(cov).min() < -0.1  # the unconstrained GMV really does short

    w_lo = minimum_variance_weights(cov, PortfolioConstraints(long_only=True))
    mu = _implied_shrinkage(cov, w_lo)
    cov_tilde = cov - (np.outer(mu, np.ones_like(mu)) + np.outer(np.ones_like(mu), mu))
    w_recovered = min_variance_weights(cov_tilde)

    assert np.max(np.abs(w_recovered - w_lo)) < 1e-10


def test_kkt_multiplier_is_nonnegative_and_complementary() -> None:
    """μ ≥ 0 and μ_i·w_i = 0 — the constraint's shadow price, recovered from the primal."""
    cov = _ill_conditioned_sample_cov(10, 40, seed=3)
    w_lo = minimum_variance_weights(cov, PortfolioConstraints(long_only=True))
    mu = _implied_shrinkage(cov, w_lo)
    assert np.all(mu >= -1e-9)  # dual feasibility
    assert np.max(np.abs(mu * w_lo)) < 1e-10  # complementary slackness


def test_shrinkage_targets_the_assets_the_unconstrained_gmv_shorts() -> None:
    """The shrunk assets (μ>0) are exactly those held at the zero bound (would be shorted)."""
    cov = _ill_conditioned_sample_cov(10, 40, seed=3)
    w_lo = minimum_variance_weights(cov, PortfolioConstraints(long_only=True))
    mu = _implied_shrinkage(cov, w_lo)
    shrunk = mu > 1e-8
    at_bound = w_lo < 1e-6
    assert np.array_equal(shrunk, at_bound)
    assert shrunk.any()  # at least one binding constraint (otherwise the claim is vacuous)


# --------------------------------------------------------------------------- #
# The practical consequence (out of sample, synthetic → CI-safe)
# --------------------------------------------------------------------------- #


def _realized_oos_vol(weights: np.ndarray, test_returns: np.ndarray) -> float:
    return float(np.std(test_returns @ weights, ddof=1))


def test_long_only_regularizes_the_sample_covariance_out_of_sample() -> None:
    """Long-only rescues the sample covariance and closes the gap to shrinkage.

    On a synthetic factor panel in the ill-conditioned regime (assets close to the
    training length), the unconstrained sample-covariance GMV realises far higher OOS
    volatility than the *same* estimator constrained long-only — and long-only sample
    then sits right next to long-only Ledoit-Wolf. The constraint did the regularising.
    """
    rng = np.random.default_rng(0)
    n_assets, train, test = 30, 40, 240
    betas = rng.uniform(0.3, 1.5, size=(n_assets, 4))
    data = generate_factor_data(train + test, betas, rng, specific_vols=0.06)
    r_train, r_test = data.asset_returns[:train], data.asset_returns[train:]

    sample = SampleCovariance().estimate(r_train)
    lw = LedoitWolfCovariance().estimate(r_train)
    long_only = PortfolioConstraints(long_only=True)

    vol_uncon_sample = _realized_oos_vol(min_variance_weights(sample), r_test)
    vol_lo_sample = _realized_oos_vol(minimum_variance_weights(sample, long_only), r_test)
    vol_lo_lw = _realized_oos_vol(minimum_variance_weights(lw, long_only), r_test)

    # The constraint roughly halves realised risk on the sample covariance...
    assert vol_lo_sample < 0.75 * vol_uncon_sample
    # ...and closes the gap to shrinkage (long-only sample ≈ long-only LW).
    assert abs(vol_lo_sample - vol_lo_lw) < 0.3 * vol_lo_sample
