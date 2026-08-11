r"""Log-GARCH(1,1) — the Type-II EGARCH-family (EGF) model, on the Phase-0 QMLE engine (Phase 2).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (WP 2026-04 §2.1 representations (10)-(13) + App. C.3; the EGF papers Ayensu et al. 2026 / Peitz
    et al. 2026; Log-GARCH of Geweke 1986 / Milhøj 1987 / Pantula 1986), **never** the `fEGarch`
    source. Validated against committed `fEGarch` *output* fixtures.

Log-GARCH is the **Type-II** EGF model: instead of Type-I's explicit asymmetry transformation, the
log-variance is an ARMA in the **log-square innovation** :math:`\xi_t = \ln(\eta_t^2) -
\operatorname{E}[\ln \eta_t^2]`. In the package's representation (10)-(11), for orders ``(1, 1)``
this reduces to the ARMA form (WP 2026-04 Eq. 13)

.. math::

    r_t = \mu + \sigma_t\,\eta_t,\qquad
    \ln\sigma_t^2 = \omega + \phi_1\,\ln\sigma_{t-1}^2 + (\psi_1 + \phi_1)\,\xi_{t-1},

with the recursion intercept :math:`\omega = \omega_\sigma\,\phi(1) = \omega_\sigma(1 - \phi_1)`.
Note the **news-impact loading is the combined** :math:`(\psi_1 + \phi_1)`, but :math:`\phi_1` (the
AR coefficient on :math:`\ln\sigma^2`) and :math:`\psi_1` (the MA coefficient on :math:`\xi`, to lag
``q = 1`` for Type-II — which is why ``(1, 1)`` has a free :math:`\psi_1` where EGARCH did not) are
reported separately. There is **no asymmetry term** — Type-II is structurally symmetric (no
:math:`\kappa`). `fEGarch` reports :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]` as
``omega_sig``; the parameter vector is ``(mu, omega_sig, phi1, psi1)`` (+ any shape parameters).

**Log-square centering.** :math:`\xi` centers on :math:`\operatorname{E}[\ln\eta^2]` — a
*log*-moment, unlike EGARCH's :math:`\operatorname{E}|\eta|`. It is sourced from the Phase-0
distribution layer's :meth:`mean_log_sq` (``norm`` → :math:`\psi(\tfrac12) + \ln 2 = -\gamma_E -
\ln 2 \approx -1.2703628`) and passed into the recursion as a captured value — never hard-coded (the
same seam as EGARCH's :math:`\operatorname{E}|\eta|`). Only the normal is wired and validated here;
jointly-estimated-shape distributions (``std`` / ``ged``) and the skewed variants are deferred.

**Pre-sample conditioning (confirmed against the fixture, machine precision).** Per WP 2026-04
App. C.3 — exactly as for EGARCH — the pre-sample :math:`\xi` history is zero and
:math:`\ln\sigma_0^2 = \omega + \phi_1\ln\operatorname{Var}(r)` with ``ddof=1``. Reconstructing the
fixture's :math:`\sigma`-series from its reported parameters matches to ``~9e-16`` (``ddof=0`` gives
``~2e-6``).

**Near-common-root identification (a data/model property, recorded honestly).** On the committed
series the fit sits near :math:`\psi_1 \approx -\phi_1` (``phi1=0.989``, ``psi1=-0.954``), so the
ARMA polynomials nearly cancel and the loading :math:`(\psi_1 + \phi_1) \approx 0.035` is small. The
likelihood is **flat/multimodal along the** :math:`\phi_1 \approx -\psi_1` **ridge**: the
conditional variance, the combined loading and the log-likelihood are pinned, but :math:`\phi_1` and
:math:`\psi_1` *individually* are weakly identified — a different start finds a marginally
higher-likelihood optimum. We start at ``phi1=0.95, psi1=-0.9`` (the regime Log-GARCH occupies) to
land in fEGarch's basin. This is WP 2026-04 §6's "least-stable family member" behaviour — a property
of this data/model, not a port defect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "fit_loggarch",
    "loggarch_recursion",
    "loggarch_sim",
]

_VAR_NAMES = ("mu", "omega_sig", "phi1", "psi1")
_MIN_ETA_SQ = 1e-300  # floor on eta^2 in ln(eta^2) — Log-GARCH needs eta != 0 a.s.


def loggarch_recursion(
    params: FloatArray, returns: FloatArray, *, mean_log_sq: float
) -> FloatArray:
    r"""Log-GARCH(1,1) conditional variance :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    Parameters
    ----------
    params : ndarray, shape (4,)
        ``(mu, omega_sig, phi1, psi1)`` — ``omega_sig`` is
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]` (the recursion intercept
        :math:`\omega = \omega_\sigma(1 - \phi_1)` and the loading :math:`(\psi_1 + \phi_1)` are
        formed internally).
    returns : ndarray, shape (T,)
        The return series.
    mean_log_sq : float
        :math:`\operatorname{E}[\ln\eta^2]`, the log-square moment of the standardized innovation,
        from the conditional distribution's :meth:`mean_log_sq` (keyword-only).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`, seeded with
        :math:`\ln\sigma_0^2 = \omega + \phi_1\ln\operatorname{Var}(r)` (``ddof=1``) and a zero
        pre-sample :math:`\xi` history.
    """
    mu, omega_sig, phi1, psi1 = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    n = y.size
    omega = omega_sig * (1.0 - phi1)  # recursion intercept omega = omega_sig * phi(1)
    xi_loading = psi1 + phi1  # combined ARMA news-impact loading on xi
    log_var = np.empty(n, dtype=np.float64)
    log_var[0] = omega + phi1 * np.log(np.var(y, ddof=1))  # xi pre-sample = 0
    for t in range(1, n):
        eta = resid[t - 1] / np.exp(log_var[t - 1] / 2.0)
        # Log-GARCH requires eta != 0 a.s.; floor eta^2 so a chance zero residual gives a large
        # finite xi (the engine then rejects the point) rather than a -inf warning.
        xi = np.log(max(eta * eta, _MIN_ETA_SQ)) - mean_log_sq
        log_var[t] = omega + phi1 * log_var[t - 1] + xi_loading * xi
    return np.exp(log_var)


def fit_loggarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit Log-GARCH(1,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``). Only ``"norm"`` is
        validated in this phase (see the module note on the distribution seam).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, psi1`` (+ shape parameters), log-likelihood,
        per-observation AIC/BIC, and the conditional-volatility series (original return units).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    # E[ln eta^2] from the distribution layer (norm -> -gamma_E - ln2); captured, not hard-coded.
    mean_log_sq = distribution.mean_log_sq(distribution.param_start)

    def recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
        return loggarch_recursion(params, returns, mean_log_sq=mean_log_sq)

    # Start near the flat phi1 ~ -psi1 ridge (high AR persistence, strong negative MA — the regime
    # Log-GARCH typically occupies) to land the optimizer in fEGarch's basin on this stiff surface.
    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), 0.95, -0.9)
    var_bounds = ((-10.0, 10.0), (-50.0, 50.0), (-0.9999, 0.9999), (-0.9999, 0.9999))

    # Near-common-root ridge: the gradient-based default (L-BFGS-B) stalls on the flat phi1 ~ -psi1
    # surface, so Log-GARCH uses the derivative-free Nelder-Mead simplex with tight tolerances.
    result = quasi_max_likelihood(
        scaled,
        recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=_VAR_NAMES,
        mean=True,
        method="Nelder-Mead",
        options={"maxiter": 8000, "xatol": 1e-10, "fatol": 1e-10},
    )

    # Undo the scaling: mu is multiplicative (~scale); omega_sig is a log-variance intercept, so it
    # shifts ADDITIVELY by ln(scale^2); phi1/psi1 and shape parameters are scale-invariant.
    names = result.param_names
    values = np.asarray(result.params, dtype=np.float64).copy()
    std_errors_arr = np.asarray(result.std_errors, dtype=np.float64).copy()
    values[0] *= scale
    values[1] += np.log(scale**2)
    std_errors_arr[0] *= scale  # additive shift on omega_sig leaves its SE unchanged
    params = {name: float(v) for name, v in zip(names, values, strict=True)}
    std_errors = {name: float(s) for name, s in zip(names, std_errors_arr, strict=True)}

    loglik = result.loglikelihood - n * np.log(scale)  # Jacobian of the rescaling
    k = result.params.size
    aic = (2.0 * k - 2.0 * loglik) / n
    bic = (k * np.log(n) - 2.0 * loglik) / n
    conditional_volatility = np.sqrt(result.conditional_variance) * scale

    return GarchFit(
        cond_dist=cond_dist,
        params=params,
        std_errors=std_errors,
        loglikelihood=float(loglik),
        aic=float(aic),
        bic=float(bic),
        conditional_volatility=np.asarray(conditional_volatility, dtype=np.float64),
        n_obs=n,
        converged=result.converged,
    )


def loggarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega_sig: float,
    phi1: float,
    psi1: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a Log-GARCH(1,1) process under a chosen conditional distribution.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig : float
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`, the unconditional log-variance.
    phi1, psi1 : float
        Log-GARCH parameters; requires ``|phi1| < 1`` for stationarity of the log-variance. The
        news-impact loading is the combined :math:`(\psi_1 + \phi_1)`.
    cond_dist : str, optional
        Conditional-distribution code for the standardized innovations (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution (e.g. ``(nu,)`` for ``std``).
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded so the recursion forgets its start (default 500).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)`` — the simulated returns and conditional SDs.

    Raises
    ------
    ValueError
        If ``n`` is not positive or ``|phi1| >= 1``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not abs(phi1) < 1.0:
        raise ValueError("require |phi1| < 1 for a stationary Log-GARCH(1,1)")

    distribution = get_distribution(cond_dist)
    mean_log_sq = distribution.mean_log_sq(dist_params)
    omega = omega_sig * (1.0 - phi1)
    xi_loading = psi1 + phi1
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    log_var = np.empty(total, dtype=np.float64)
    log_var[0] = omega_sig  # start at the unconditional mean E[ln sigma^2]
    for t in range(1, total):
        xi = np.log(eta[t - 1] * eta[t - 1]) - mean_log_sq
        log_var[t] = omega + phi1 * log_var[t - 1] + xi_loading * xi
    sigma = np.exp(log_var / 2.0)
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )
