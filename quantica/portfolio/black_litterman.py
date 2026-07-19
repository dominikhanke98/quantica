r"""Black--Litterman (1992) — blending market equilibrium with subjective views.

Naive Markowitz is famously unstable: it treats noisy expected-return estimates as
truth, and because it inverts the covariance, tiny changes in those estimates swing the
optimal weights wildly (and pile into a few names). Black--Litterman fixes this by
**shrinking toward a market-implied equilibrium**:

1. **Reverse optimisation** — the market-cap (or benchmark) weights are, by revealed
   preference, the optimal holding of a representative investor, so they imply
   *equilibrium* excess returns :math:`\pi = \delta\,\Sigma\,w_{\text{mkt}}` (invert the
   mean-variance first-order condition for :math:`\mu` given the weights).
2. **Views** — subjective forecasts are stated as :math:`P\mu = Q + \varepsilon`,
   :math:`\varepsilon \sim \mathcal N(0, \Omega)`: a picking matrix :math:`P` (each row a
   portfolio the view is about), a target vector :math:`Q`, and an uncertainty
   :math:`\Omega`. The default :math:`\Omega = \operatorname{diag}(P\,\tau\Sigma\,P^\top)`
   scales each view's uncertainty to the prior (He & Litterman, 1999).
3. **Posterior** — the Bayesian blend of the equilibrium prior and the views (the BL
   *master formula*),

   .. math::

       \mu_{BL} = \big[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\big]^{-1}
                  \big[(\tau\Sigma)^{-1}\pi + P^\top\Omega^{-1}Q\big],
       \qquad
       \Sigma_{BL} = \Sigma + \big[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\big]^{-1},

   which feeds :func:`~quantica.portfolio.construction.mean_variance_weights`.

Because the posterior only nudges the equilibrium in the direction of the views (and
by an amount set by their confidence), the resulting weights are stable and diversified
— the classic reason BL is the workhorse of institutional asset allocation. With **no
views** the posterior returns collapse to the equilibrium, a clean known-truth anchor.

References
----------
Black, F. & Litterman, R. (1992), "Global portfolio optimization", *Financial Analysts
Journal*. He, G. & Litterman, R. (1999), "The intuition behind Black--Litterman model
portfolios", Goldman Sachs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "BlackLittermanResult",
    "black_litterman",
    "implied_equilibrium_returns",
]

_DEFAULT_TAU = 0.05  # scale of the prior uncertainty in the equilibrium mean


def _as_cov(cov: FloatArray) -> FloatArray:
    sigma = np.asarray(cov, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be a square 2-D matrix, got shape {sigma.shape}")
    return sigma


@dataclass(frozen=True)
class BlackLittermanResult:
    """The Black--Litterman posterior and the equilibrium prior it was built from.

    Attributes
    ----------
    posterior_returns : ndarray, shape (n,)
        The blended expected returns :math:`\\mu_{BL}` to feed mean-variance construction.
    posterior_cov : ndarray, shape (n, n)
        The posterior covariance :math:`\\Sigma_{BL}` (the asset covariance plus the
        posterior uncertainty in the mean).
    equilibrium_returns : ndarray, shape (n,)
        The reverse-optimised equilibrium excess returns :math:`\\pi` (the prior mean).
    """

    posterior_returns: FloatArray
    posterior_cov: FloatArray
    equilibrium_returns: FloatArray


def implied_equilibrium_returns(
    cov: FloatArray, market_weights: FloatArray, risk_aversion: float
) -> FloatArray:
    r"""Reverse-optimised equilibrium excess returns :math:`\pi = \delta\,\Sigma\,w_{\text{mkt}}`.

    Inverts the mean-variance first-order condition: if ``market_weights`` are the
    optimal holding at risk-aversion ``risk_aversion``, they imply these expected
    returns. Feeding ``pi`` back into mean-variance at the same risk-aversion recovers
    ``market_weights`` (the round-trip anchor).

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        The asset covariance :math:`\Sigma`.
    market_weights : ndarray, shape (n,)
        The market-cap or benchmark weights :math:`w_{\text{mkt}}`.
    risk_aversion : float
        The representative investor's risk-aversion :math:`\delta` (must be positive).

    Returns
    -------
    ndarray, shape (n,)
        The equilibrium excess returns :math:`\pi`.
    """
    if risk_aversion <= 0.0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    sigma = _as_cov(cov)
    w = np.asarray(market_weights, dtype=np.float64)
    if w.shape != (sigma.shape[0],):
        raise ValueError(f"market_weights must have shape ({sigma.shape[0]},), got {w.shape}")
    return np.asarray(risk_aversion * (sigma @ w), dtype=np.float64)


def black_litterman(
    cov: FloatArray,
    market_weights: FloatArray,
    risk_aversion: float,
    *,
    views_p: FloatArray | None = None,
    views_q: FloatArray | None = None,
    view_uncertainty: FloatArray | None = None,
    tau: float = _DEFAULT_TAU,
) -> BlackLittermanResult:
    r"""Black--Litterman posterior returns and covariance from equilibrium + views.

    Builds the equilibrium prior :math:`\pi = \delta\Sigma w_{\text{mkt}}`, then blends
    it with the views :math:`P\mu = Q + \varepsilon` via the master formula (see the
    module docstring). With no views (``views_p`` / ``views_q`` ``None`` or empty) the
    posterior returns collapse to the equilibrium.

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        The asset covariance :math:`\Sigma`.
    market_weights : ndarray, shape (n,)
        The market/benchmark weights, reverse-optimised into the equilibrium prior.
    risk_aversion : float
        The risk-aversion :math:`\delta` used in the reverse optimisation.
    views_p : ndarray, shape (k, n), optional
        The view picking matrix :math:`P` (each row a portfolio the view is about).
    views_q : ndarray, shape (k,), optional
        The view target returns :math:`Q`.
    view_uncertainty : ndarray, shape (k, k), optional
        The view covariance :math:`\Omega`. Defaults to the He--Litterman
        :math:`\operatorname{diag}(P\,\tau\Sigma\,P^\top)`.
    tau : float, optional
        The prior-uncertainty scale :math:`\tau` (default 0.05).

    Returns
    -------
    BlackLittermanResult
        The posterior returns, posterior covariance, and equilibrium prior.

    Raises
    ------
    ValueError
        If ``tau`` is not positive, or the view arrays are shaped inconsistently.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")
    sigma = _as_cov(cov)
    n = sigma.shape[0]
    pi = implied_equilibrium_returns(sigma, market_weights, risk_aversion)

    no_views = views_p is None or views_q is None or np.asarray(views_q, dtype=np.float64).size == 0
    if no_views:
        # The posterior of a Bayesian update with no data is the prior: equilibrium.
        return BlackLittermanResult(
            posterior_returns=pi, posterior_cov=sigma, equilibrium_returns=pi
        )

    p = np.atleast_2d(np.asarray(views_p, dtype=np.float64))
    q = np.asarray(views_q, dtype=np.float64).reshape(-1)
    if p.shape[1] != n:
        raise ValueError(f"views_p must have {n} columns, got shape {p.shape}")
    if p.shape[0] != q.shape[0]:
        raise ValueError(f"views_p rows ({p.shape[0]}) must match views_q ({q.shape[0]})")

    tau_sigma = tau * sigma
    if view_uncertainty is None:
        omega = np.diag(np.diag(p @ tau_sigma @ p.T))
    else:
        omega = np.asarray(view_uncertainty, dtype=np.float64)
        if omega.shape != (q.shape[0], q.shape[0]):
            raise ValueError(
                f"view_uncertainty must be ({q.shape[0]}, {q.shape[0]}), got {omega.shape}"
            )

    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(omega)
    posterior_precision = tau_sigma_inv + p.T @ omega_inv @ p
    rhs = tau_sigma_inv @ pi + p.T @ omega_inv @ q
    posterior_returns = np.linalg.solve(posterior_precision, rhs)
    posterior_cov = sigma + np.linalg.inv(posterior_precision)

    return BlackLittermanResult(
        posterior_returns=np.asarray(posterior_returns, dtype=np.float64),
        posterior_cov=np.asarray(posterior_cov, dtype=np.float64),
        equilibrium_returns=pi,
    )
