r"""Hierarchical Risk Parity (López de Prado, 2016) — allocation without inversion.

Markowitz optimisation inverts the covariance, and on an ill-conditioned universe
(many assets, few observations) that inversion amplifies estimation error into wild,
concentrated weights — Michaud's "error maximiser", the failure the factor step
(:mod:`quantica.factor.evaluation`) measures and Jagannathan--Ma explains. **HRP
sidesteps inversion entirely**, which is the whole point: it never solves a linear
system in :math:`\Sigma`, so there is no near-singular matrix to blow up.

Three stages (each leaning on ``scipy`` for the clustering plumbing, CLAUDE.md §3):

1. **Tree clustering** — turn the correlation matrix into a distance
   :math:`d_{ij} = \sqrt{\tfrac12 (1 - \rho_{ij})}` and build a hierarchical
   linkage tree (``scipy.cluster.hierarchy.linkage``): similar assets join low in
   the tree.
2. **Quasi-diagonalisation** — reorder the assets by the tree's leaf order
   (``scipy.cluster.hierarchy.leaves_list``) so the covariance's large entries sit
   near the diagonal and similar assets are adjacent.
3. **Recursive bisection** — walk down the ordered tree splitting each cluster in
   two and dividing its weight between the halves **inversely to their variance**
   (an inverse-variance portfolio *within* each cluster). Only cluster variances and
   diagonal inverse-variance weights are used — never a full matrix inverse.

The result is a long-only, fully-invested portfolio that is far more robust
out-of-sample than the inverting minimum-variance portfolio exactly where the sample
covariance is worst (see ``scripts/hrp_robustness_report.py``).

References
----------
López de Prado, M. (2016), "Building diversified portfolios that outperform out of
sample", *Journal of Portfolio Management* 42(4). Also *Advances in Financial Machine
Learning* (2018), ch. 16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from quantica.core.types import FloatArray, IntArray

__all__ = [
    "hrp_weights",
    "quasi_diagonal_order",
]

_DEFAULT_LINKAGE = "single"  # López de Prado's original single-linkage choice


def _as_cov(cov: FloatArray) -> FloatArray:
    sigma = np.asarray(cov, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be a square 2-D matrix, got shape {sigma.shape}")
    return sigma


def _correlation_from_cov(cov: FloatArray) -> FloatArray:
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    # Guard tiny numerical overshoot past +/-1 (would break the sqrt distance).
    return np.asarray(np.clip(corr, -1.0, 1.0), dtype=np.float64)


def quasi_diagonal_order(cov: FloatArray, linkage_method: str = _DEFAULT_LINKAGE) -> IntArray:
    r"""Return the hierarchical-clustering leaf order of the assets.

    Clusters the assets on the correlation distance
    :math:`d_{ij} = \sqrt{\tfrac12(1 - \rho_{ij})}` and returns the linkage tree's leaf
    order — the permutation that places similar assets adjacent (quasi-diagonalising the
    covariance). This is stage 1--2 of HRP, exposed for inspection and testing.

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        The asset covariance (its implied correlation drives the clustering).
    linkage_method : str, optional
        The ``scipy.cluster.hierarchy.linkage`` method (default ``"single"``, López de
        Prado's original choice).

    Returns
    -------
    ndarray of int, shape (n,)
        The asset indices in quasi-diagonal (leaf) order.
    """
    sigma = _as_cov(cov)
    if sigma.shape[0] == 1:
        return np.array([0], dtype=np.intp)
    corr = _correlation_from_cov(sigma)
    distance = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method=linkage_method)
    return np.asarray(leaves_list(tree), dtype=np.intp)


def _cluster_variance(cov: FloatArray, items: list[int]) -> float:
    """Variance of the inverse-variance portfolio on the sub-covariance ``items``.

    Uses only the diagonal (inverse-variance weights) and the sub-block — no inverse of
    the full covariance is ever formed.
    """
    sub = cov[np.ix_(items, items)]
    inv_var = 1.0 / np.diag(sub)
    weights = inv_var / inv_var.sum()
    return float(weights @ sub @ weights)


def hrp_weights(cov: FloatArray, linkage_method: str = _DEFAULT_LINKAGE) -> FloatArray:
    r"""Hierarchical Risk Parity weights — long-only, fully invested, no inversion.

    Runs the three HRP stages (cluster the correlation tree, reorder by leaf order,
    recursively bisect the ordered tree splitting weight inversely to cluster variance)
    and returns the allocation. The covariance is consumed as produced by a
    :class:`~quantica.factor.estimators.CovarianceEstimator`; unlike
    :func:`~quantica.portfolio.construction.minimum_variance_weights` it never inverts
    it, so a near-singular :math:`\Sigma` does not blow the weights up.

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        The asset covariance.
    linkage_method : str, optional
        The ``scipy`` linkage method for the clustering (default ``"single"``).

    Returns
    -------
    ndarray, shape (n,)
        Non-negative weights summing to one.
    """
    sigma = _as_cov(cov)
    n = sigma.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64)

    order = quasi_diagonal_order(sigma, linkage_method)
    weights = np.ones(n, dtype=np.float64)
    clusters: list[list[int]] = [order.tolist()]
    while clusters:
        next_clusters: list[list[int]] = []
        for items in clusters:
            if len(items) <= 1:
                continue
            split = len(items) // 2
            left, right = items[:split], items[split:]
            var_left = _cluster_variance(sigma, left)
            var_right = _cluster_variance(sigma, right)
            # Give the lower-variance cluster the larger share (inverse-variance split).
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters

    return np.asarray(weights / weights.sum(), dtype=np.float64)
