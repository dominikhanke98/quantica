r"""Vector Error Correction Models — the multivariate generalisation of pairwise cointegration.

A VECM describes an :math:`n`-dimensional system that is individually :math:`I(1)` (each series
random-walk-like) but tied together by :math:`r` stationary long-run relationships — the
multivariate version of the pairwise cointegration the stat-arb pillar builds for :math:`n = 2`.
In first differences,

.. math::

    \Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{k} \Gamma_i \Delta y_{t-i} + \mu + \varepsilon_t,
    \qquad \Pi = \alpha\beta',

where the columns of :math:`\beta` (``n x r``) are the **cointegrating vectors** (the combinations
that are stationary) and :math:`\alpha` (``n x r``) holds the **adjustment speeds** (how each series
error-corrects back toward the long-run relations).

The estimation is Johansen's reduced-rank regression, hand-implemented here
(:func:`fit_vecm`): concentrate out the lagged differences, form the moment matrices, and solve the
eigenvalue problem whose leading eigenvectors are :math:`\beta`. The cointegrating **rank** is
selected by the Johansen trace test (reusing :func:`quantica.statarb.johansen`), and for
:math:`n = 2, r = 1` the single cointegrating vector reduces to the pairwise hedge ratio the
stat-arb pillar estimates — the reduction anchor. Coefficients match ``statsmodels``' ``VECM``.

References
----------
Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector
Autoregressive Models." *Econometrica* 59(6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.statarb import johansen

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "VecmResult",
    "fit_vecm",
    "select_cointegration_rank",
]


def select_cointegration_rank(
    data: FloatArray, *, k_ar_diff: int = 1, det_order: int = 0, significance: float = 0.05
) -> int:
    """Select the cointegration rank by the Johansen trace test (reuses the stat-arb Johansen).

    Parameters
    ----------
    data : ndarray, shape (T, n)
        The level series (columns are the individual :math:`I(1)` series).
    k_ar_diff : int, optional
        Number of lagged differences in the VECM (default 1).
    det_order : int, optional
        Deterministic assumption passed to Johansen: ``-1`` none, ``0`` constant, ``1`` trend
        (default 0).
    significance : {0.10, 0.05, 0.01}, optional
        Test level (default 0.05).

    Returns
    -------
    int
        The estimated number of cointegrating relations, ``0 .. n``.
    """
    result = johansen(np.asarray(data, dtype=np.float64), det_order=det_order, k_ar_diff=k_ar_diff)
    return result.rank(significance, statistic="trace")


@dataclass(frozen=True)
class VecmResult:
    r"""A fitted Vector Error Correction Model.

    Attributes
    ----------
    rank : int
        The cointegration rank :math:`r` used for the fit.
    alpha : ndarray, shape (n, r)
        Adjustment/loading matrix — how fast each series corrects toward the long-run relations.
    beta : ndarray, shape (n, r)
        Cointegrating vectors (columns), normalised so the leading ``r x r`` block is the identity
        (Phillips normalisation), matching ``statsmodels``.
    gamma : ndarray, shape (n, n*k)
        Short-run dynamics: the stacked :math:`[\Gamma_1, \dots, \Gamma_k]` coefficient matrices.
    intercept : ndarray, shape (n,)
        The fitted constant (zeros when ``deterministic="n"``).
    eigenvalues : ndarray, shape (n,)
        The Johansen eigenvalues (descending); the leading ``r`` drive the cointegration.
    k_ar_diff : int
        Number of lagged differences used.
    """

    rank: int
    alpha: FloatArray
    beta: FloatArray
    gamma: FloatArray
    intercept: FloatArray
    eigenvalues: FloatArray
    k_ar_diff: int

    @property
    def long_run_matrix(self) -> FloatArray:
        r"""The long-run impact matrix :math:`\Pi = \alpha\beta'` (normalisation-invariant)."""
        return np.asarray(self.alpha @ self.beta.T, dtype=np.float64)

    def hedge_ratio(self) -> float:
        r"""For a bivariate rank-1 system, the pairwise hedge ratio :math:`-\beta_2/\beta_1`.

        With ``beta`` Phillips-normalised (``beta[0] = 1``) this is ``-beta[1, 0]`` — the same
        cointegrating hedge ratio the stat-arb pillar estimates for a pair, the reduction anchor.

        Returns
        -------
        float
            The hedge ratio of the second series on the first.

        Raises
        ------
        ValueError
            If the system is not bivariate with rank 1.
        """
        if self.beta.shape != (2, 1):
            raise ValueError("hedge_ratio is only defined for a bivariate rank-1 system")
        return float(-self.beta[1, 0])


def _lagged_difference_regressors(diffs: FloatArray, k: int, n_rows: int) -> FloatArray:
    """Stack the ``k`` lagged difference blocks aligned to the current differences."""
    if k == 0:
        return np.empty((n_rows, 0), dtype=np.float64)
    blocks = [diffs[k - 1 - i : k - 1 - i + n_rows] for i in range(k)]
    return np.hstack(blocks)


def fit_vecm(
    data: FloatArray, *, rank: int, k_ar_diff: int = 1, deterministic: str = "co"
) -> VecmResult:
    r"""Fit a VECM by Johansen reduced-rank regression (hand-implemented).

    Estimates :math:`\alpha`, :math:`\beta`, the short-run :math:`\Gamma_i` and an optional constant
    for a system of cointegration ``rank``. The cointegrating vectors are the leading eigenvectors
    of the reduced-rank problem; the loadings and short-run terms follow in closed form.

    Parameters
    ----------
    data : ndarray, shape (T, n)
        The level series (columns are the series).
    rank : int
        The cointegration rank :math:`r` (``0 < r < n``); typically from
        :func:`select_cointegration_rank`.
    k_ar_diff : int, optional
        Number of lagged differences :math:`k` in the VECM (default 1).
    deterministic : {"co", "n"}, optional
        ``"co"`` includes a constant inside the short-run dynamics (default); ``"n"`` excludes it.

    Returns
    -------
    VecmResult
        The fitted matrices and eigenvalues.

    Raises
    ------
    ValueError
        If ``rank`` is not in ``1 .. n-1`` or there are too few observations.
    """
    y = np.asarray(data, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("data must be 2-D (T, n)")
    _n_obs, n = y.shape
    if not 0 < rank < n:
        raise ValueError(f"rank must be in 1..{n - 1}, got {rank}")

    diffs = np.diff(y, axis=0)  # Δy, shape (T-1, n)
    k = k_ar_diff
    n_rows = diffs.shape[0] - k
    if n_rows <= n:
        raise ValueError("too few observations for the requested k_ar_diff")

    z0 = diffs[k:]  # current differences Δy_t
    z1 = y[k:-1]  # lagged levels y_{t-1}
    z2 = _lagged_difference_regressors(diffs, k, n_rows)
    if deterministic == "co":
        z2 = np.hstack([z2, np.ones((n_rows, 1))])
    elif deterministic != "n":
        raise ValueError("deterministic must be 'co' or 'n'")

    # Concentrate out the short-run regressors z2 from z0 and z1.
    if z2.shape[1] > 0:
        projector = z2 @ np.linalg.solve(z2.T @ z2, z2.T)
        r0 = z0 - projector @ z0
        r1 = z1 - projector @ z1
    else:
        r0, r1 = z0, z1

    t_eff = r0.shape[0]
    s00 = r0.T @ r0 / t_eff
    s11 = r1.T @ r1 / t_eff
    s01 = r0.T @ r1 / t_eff

    # Reduced-rank eigenproblem: eigenvectors of S11^-1 S10 S00^-1 S01 span the cointegration space.
    eig_matrix = np.linalg.solve(s11, s01.T) @ np.linalg.solve(s00, s01)
    eigenvalues, eigenvectors = np.linalg.eig(eig_matrix)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    beta_raw = eigenvectors[:, :rank]
    # Phillips normalisation: leading r x r block = identity (matches statsmodels).
    beta = beta_raw @ np.linalg.inv(beta_raw[:rank, :])
    alpha = s01 @ beta @ np.linalg.inv(beta.T @ s11 @ beta)

    # Short-run Gamma and constant: OLS of (dy - alpha beta' y_{t-1}) on the lagged diffs.
    corrected = z0 - z1 @ beta @ alpha.T
    if z2.shape[1] > 0:
        coefficients = np.linalg.solve(z2.T @ z2, z2.T @ corrected)  # (p, n)
    else:
        coefficients = np.empty((0, n), dtype=np.float64)
    if deterministic == "co":
        intercept = coefficients[-1]
        gamma = coefficients[:-1].T
    else:
        intercept = np.zeros(n, dtype=np.float64)
        gamma = coefficients.T

    return VecmResult(
        rank=rank,
        alpha=np.asarray(alpha, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        gamma=np.asarray(gamma, dtype=np.float64),
        intercept=np.asarray(intercept, dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        k_ar_diff=k,
    )
