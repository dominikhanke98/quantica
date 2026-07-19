"""Validation of Hierarchical Risk Parity (numerical-validation skill).

Structural checks pin the allocation (weights sum to one, long-only, the two-asset case
reduces to the inverse-variance closed form, equal-variance uncorrelated assets get
equal weight, planted cluster blocks are recovered by the tree). The **headline** is the
out-of-sample robustness tie-back to factor stage 2 / Jagannathan-Ma: on an
ill-conditioned universe (assets close to observations) the inverting minimum-variance
portfolio blows up while HRP — which never inverts the covariance — stays bounded.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.factor.data import generate_factor_data
from quantica.factor.estimators import SampleCovariance, condition_number, min_variance_weights
from quantica.portfolio.hrp import hrp_weights, quasi_diagonal_order


def _sample_cov(returns: np.ndarray) -> np.ndarray:
    return np.cov(np.asarray(returns, dtype=np.float64), rowvar=False)


# --------------------------------------------------------------------------- #
# Structural correctness
# --------------------------------------------------------------------------- #


def test_weights_sum_to_one_and_long_only() -> None:
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((200, 12)) @ rng.standard_normal((12, 12)) * 0.05
    w = hrp_weights(_sample_cov(returns))
    assert np.isclose(w.sum(), 1.0, atol=1e-12)
    assert w.min() >= 0.0


def test_two_assets_reduce_to_inverse_variance() -> None:
    # On two assets HRP splits weight inversely to variance — a closed-form anchor.
    cov = np.diag([0.01, 0.04])
    w = hrp_weights(cov)
    inv_var = np.array([1.0 / 0.01, 1.0 / 0.04])
    assert np.allclose(w, inv_var / inv_var.sum(), atol=1e-12)  # [0.8, 0.2]


def test_equal_variance_uncorrelated_is_equal_weight() -> None:
    w = hrp_weights(np.eye(6))
    assert np.allclose(w, np.full(6, 1.0 / 6.0), atol=1e-12)


def test_recovers_planted_cluster_blocks() -> None:
    # Two independent blocks of correlated assets: the tree must place each block
    # contiguously in its leaf order.
    rng = np.random.default_rng(3)
    n = 10
    loadings = np.zeros((n, 3))
    loadings[:5, 0] = 1.0  # block A factor
    loadings[5:, 1] = 1.0  # block B factor
    loadings[:, 2] = 0.15  # weak common factor
    factors = rng.standard_normal((400, 3))
    noise = rng.standard_normal((400, n)) * 0.5
    cov = _sample_cov(factors @ loadings.T + noise)

    order = quasi_diagonal_order(cov).tolist()
    for block in (list(range(5)), list(range(5, 10))):
        positions = sorted(order.index(i) for i in block)
        assert positions == list(range(positions[0], positions[0] + len(block)))


def test_rejects_non_square_cov() -> None:
    with pytest.raises(ValueError, match="square"):
        hrp_weights(np.ones((3, 4)))


# --------------------------------------------------------------------------- #
# The headline: out-of-sample robustness where min-variance inverts and blows up
# --------------------------------------------------------------------------- #


def _realized_vol(weights: np.ndarray, test_returns: np.ndarray) -> float:
    return float(np.std(test_returns @ weights, ddof=1))


def test_hrp_is_robust_where_min_variance_blows_up() -> None:
    """On an ill-conditioned universe HRP realises far lower OOS vol than min-variance.

    Assets (45) are close to the training length (48), so the sample covariance is
    near-singular (condition number >> 1) and the *unconstrained* minimum-variance
    portfolio — which inverts it — produces wild, heavily-levered weights that realise
    huge out-of-sample volatility. HRP never inverts the covariance, so it stays bounded
    and long-only. Direct tie-back to the factor-stage-2 error-maximiser finding
    (measured here ~3x worse OOS vol, ~20x the leverage).
    """
    rng = np.random.default_rng(0)
    n_assets, train, test = 45, 48, 250  # n/T ~ 0.94: near-singular sample covariance
    betas = rng.uniform(0.3, 1.5, size=(n_assets, 4))
    data = generate_factor_data(train + test, betas, rng, specific_vols=0.06)
    r_train, r_test = data.asset_returns[:train], data.asset_returns[train:]
    cov = SampleCovariance().estimate(r_train)

    assert condition_number(cov) > 1e3  # genuinely ill-conditioned

    vol_min_var = _realized_vol(min_variance_weights(cov), r_test)  # inverts Sigma
    vol_hrp = _realized_vol(hrp_weights(cov), r_test)  # never inverts

    # HRP realises materially lower out-of-sample volatility than the inverting GMV.
    assert vol_hrp < 0.5 * vol_min_var
    # And its weights are diversified (long-only, no wild leverage), unlike the GMV.
    w_hrp = hrp_weights(cov)
    assert w_hrp.min() >= 0.0
    assert np.abs(min_variance_weights(cov)).sum() > 3.0 * np.abs(w_hrp).sum()
