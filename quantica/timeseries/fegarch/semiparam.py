r"""Local-polynomial regression — the semiparametric scale-estimation core (Phase 7, sub-build 1).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Feng, Gries & Fritz 2020; Beran & Feng 2002), **never** the `smoots` / `esemifar` source (their
    Rcpp/C++ is the reference implementation, treated exactly as `fEGarch` is). Validated against
    the committed `smoots::gsmooth` *output* fixtures (``smooth_gsmooth_*``).

Local-polynomial regression estimates the deterministic trend :math:`m(x)` in the equidistant
nonparametric model :math:`y_t = m(x_t) + \varepsilon_t`, :math:`x_t = t/n \in (0, 1]`, at a
**given** bandwidth (the data-driven bandwidth selection is sub-build 2). It is the load-bearing
step of the semiparametric scale function: fEGarch smooths the log-squared demeaned returns
:math:`\tilde w_t = \ln[(y_t - \bar y)^2]` with this estimator to obtain the nonparametric scale.

The estimator (:func:`local_poly`), pinned machine-exact against ``smoots::gsmooth``:

* **Kernels** (``mu``): the normalized :math:`(1 - u^2)^{\mu}` family on :math:`[-1, 1]` — ``mu=0``
  uniform, ``mu=1`` Epanechnikov :math:`\tfrac34(1-u^2)`, ``mu=2`` bisquare
  :math:`\tfrac{15}{16}(1-u^2)^2`, ``mu=3`` triweight :math:`\tfrac{35}{32}(1-u^2)^3` (matching
  fEGarch's ``locpol_spec`` ``kernel_order``). The normalization cancels in the weighted least
  squares, so the unnormalized :math:`(1-u^2)^{\mu}` is used directly.
* **Estimator**: with :math:`m = \lfloor n b \rfloor` the bandwidth in points, at each :math:`t`
fit a
  degree-``p`` polynomial in the index offset :math:`d_i = i - t` by weighted least squares with
  weights :math:`K(d_i / s)`, and take the fitted intercept as :math:`\hat m(x_t)`. The kernel scale
  is :math:`s = \max_i |d_i| + 1` (so :math:`s = m + 1` in the interior; the intercept is invariant
  to rescaling the design, so the raw offset is used). ``p - v`` must be odd (here :math:`v = 0`, so
  ``p`` is odd — ``p = 1`` local-linear, ``p = 3`` local-cubic, the two ``locpol_spec`` orders).
* **Boundary** (``boundary``): near the ends the symmetric window :math:`[t-m, t+m]` runs off the
  data. ``"fixed"`` (smoots ``bb=0``) truncates it to :math:`[\max(0, t-m), \min(n-1, t+m)]` — a
  fixed bandwidth, fewer points at the boundary; ``"knn"`` (smoots ``bb=1``, the default) shifts
  the window inward to keep :math:`2m+1` points (k-nearest-neighbour), with the scale :math:`s`
  widening to the window's larger half-width. The interior rows are identical for both; only the
  boundary rows differ. (The ``locpol_spec`` ``boundary_method`` ``extend``/``shorten`` ↔
  ``fixed``/``knn`` correspondence is pinned in sub-build 3, the semiparametric wiring.)

Reconstructing the fixtures this way is machine-exact: interior ``~1e-14``, boundary ``~1e-14``
(``p=1``) to ``~1e-12`` (``p=3``, the cubic design's floating-point summation order). This
determinism matters because sub-build 2 (the iterative-plug-in bandwidth) calls this repeatedly.

**Sub-build 2 (IPI bandwidth) — the machine-exact components are built; the auto-iterator is not.**
This module adds the two further pieces that *are* cleanly clean-room-reproducible: the derivative
local polynomial :func:`local_poly` (``v = k``, for the AMISE numerator :math:`I[m^{(k)}]`, with
:func:`integrated_squared_derivative`), and the short-memory variance factor
:func:`bartlett_variance_factor` (the Bartlett lag-window, **given** the window ``M``). The full
**AMISE iterative-plug-in bandwidth selector is deliberately not built** — it depends on references
absent from ``literature/``: the equivalent-kernel constants :math:`\beta_{\nu,k}`, :math:`R(K)`,
:math:`K(0)` and the enlargement factor :math:`C_F` are tabulated in **Feng-Heiler (2009)** (cited,
not reproduced by Feng-Gries-Fritz), and the short-memory window-width ``M`` (``L0.opt``) is the
nested **Bühlmann (1996)** spectral-density IPI. The long-memory variance factor (``esemifar``) is a
FARIMA fracdiff-MLE — an *optimizer-dependent fit* reproducible only to the port's ~1e-6 fit
tolerance (not machine-exact). These limits, and the paper-vs-implementation
:math:`(M+0.5)`-vs-:math:`(M+1)` Bartlett-weight divergence, are recorded in
``docs/fegarch-spec-notes.md``.

**Sub-build 3 (the fixed-bandwidth semiparametric fit) — fully clean-room.**
:func:`semiparametric_fit` wires the six-step WP171 §2.2.1 procedure: the sample mean
:math:`\bar y`; the log-squared
:math:`\tilde w_t = \ln[(y_t-\bar y)^2]`; the local-poly trend :math:`\hat m` at a *fixed* bandwidth
(:func:`local_poly`); the **empirical** retransform correction
:math:`\hat C_T = -\ln[n^{-1}\sum e^{\tilde w_t-\hat m_t}]`; the scale
:math:`\hat s = \exp\{(\hat m-\hat C_T)/2\}` (:func:`semiparametric_scale`); the de-scaled returns
:math:`\hat r = (y-\bar y)/\hat s`; and a **zero-mean** EGF fit on :math:`\hat r` giving the total
volatility :math:`\hat\Omega = \hat s\,\tilde\sigma`. At a fixed user-supplied bandwidth the scale
steps are **machine-exact** (and route-independent — the SM/LM selector only governs *automatic*
bandwidth, bypassed here) and the parametric fit is at the established fit tolerance, so the whole
semiparametric fit is clean-room-reproducible with **no** dependency on the blocked auto-selectors.
Automatic bandwidth (``bwidth=None``) raises, pointing to the Bühlmann/Feng-Heiler gap.

References
----------
Feng, Y., Gries, T. & Fritz, M. (2020). "Data-Driven Local Polynomial for the Trend and its
Derivatives in Economic Time Series." *Journal of Nonparametric Statistics* 32(2).
Beran, J. & Feng, Y. (2002). "SEMIFAR models — a semiparametric approach to modelling trends,
long-range dependence and nonstationarity." *Computational Statistics & Data Analysis* 40(2).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING, Literal

import numpy as np

from quantica.timeseries.fegarch.asymmetric import fit_aparch, fit_gjr, fit_tgarch
from quantica.timeseries.fegarch.egarch import fit_egarch, fit_megarch, fit_mloggarch
from quantica.timeseries.fegarch.fiaparch import fit_fiaparch, fit_figjr, fit_fitgarch
from quantica.timeseries.fegarch.fiegarch import fit_fiegarch, fit_fimegarch, fit_fimloggarch
from quantica.timeseries.fegarch.figarch import fit_figarch
from quantica.timeseries.fegarch.filoggarch import fit_filoggarch
from quantica.timeseries.fegarch.garch import GarchFit, fit_garch
from quantica.timeseries.fegarch.loggarch import fit_loggarch

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantica.core.types import FloatArray

#: The 16 fEGarch inner models, keyed by their fit tag, each as a zero-mean-capable fitter. The
#: semiparametric scale is model-agnostic (steps 1-5); the inner model only enters step 6 (the
#: zero-mean QMLE on the de-scaled returns), so every fitted model composes with the same scale.
_INNER_FITTERS: dict[str, Callable[..., GarchFit]] = {
    "garch": fit_garch,
    "gjrgarch": fit_gjr,
    "tgarch": fit_tgarch,
    "aparch": fit_aparch,
    "figarch": fit_figarch,
    "figjr": fit_figjr,
    "fitgarch": fit_fitgarch,
    "fiaparch": fit_fiaparch,
    "egarch": fit_egarch,
    "megarch": fit_megarch,
    "mloggarch": fit_mloggarch,
    "loggarch": fit_loggarch,
    "fiegarch": fit_fiegarch,
    "fimegarch": fit_fimegarch,
    "fimloggarch": fit_fimloggarch,
    "filoggarch": fit_filoggarch,
}

__all__ = [
    "KERNELS",
    "SemiparametricFit",
    "bartlett_variance_factor",
    "integrated_squared_derivative",
    "local_poly",
    "semiparametric_fit",
    "semiparametric_scale",
]

#: locpol_spec boundary_method -> the gsmooth boundary rule (extend=knn/bb1, shorten=fixed/bb0).
_BOUNDARY_MAP: dict[str, Literal["fixed", "knn"]] = {"extend": "knn", "shorten": "fixed"}

#: The four second-order kernels ``locpol_spec`` exposes, keyed by the ``mu`` smoothness order.
KERNELS: dict[int, str] = {0: "uniform", 1: "epanechnikov", 2: "bisquare", 3: "triweight"}


def _kernel_weights(u: FloatArray, mu: int) -> FloatArray:
    r"""Unnormalized second-order kernel :math:`(1 - u^2)^{\mu}` on :math:`[-1, 1]` (0 outside)."""
    base = np.clip(1.0 - u * u, 0.0, None)
    if mu == 0:
        return np.asarray((base > 0.0).astype(np.float64), dtype=np.float64)
    return np.asarray(base**mu, dtype=np.float64)


def local_poly(
    y: FloatArray,
    *,
    v: int = 0,
    p: int = 3,
    mu: int = 1,
    bandwidth: float = 0.15,
    boundary: Literal["fixed", "knn"] = "knn",
) -> FloatArray:
    r"""Local-polynomial estimate of the trend or its ``v``-th derivative (``gsmooth`` core).

    Weighted-least-squares local-polynomial regression of ``y`` on the rescaled equidistant time
    grid :math:`x_t = t/n \in (0, 1]`. For ``v = 0`` this is the fitted trend :math:`\hat m(x_t)`;
    for ``v \ge 1`` it is the :math:`v`-th derivative :math:`\hat m^{(v)}(x_t)` **with respect to
    the rescaled time** — the WLS coefficient of the degree-``v`` term times :math:`v!` and times
    :math:`n^{v}` (the chain-rule factor converting the index-offset derivative to the :math:`x`
    derivative). Reproduces ``smoots::gsmooth(y, v, p, mu, b, bb)`` machine-exactly (``v = 0``); for
    ``v \ge 1`` the :math:`n^{v}` factor amplifies the ~1e-15 relative floating-point error of the
    high-order WLS solve to ~1e-9 (``v = 2``) / ~1e-3 (``v = 4``) absolute — a documented
    FP-amplification, not a convention gap.

    Parameters
    ----------
    y : ndarray, shape (n,)
        The equidistant series to smooth (e.g. the log-squared demeaned returns for the scale).
    v : int, optional
        The derivative order to estimate (0 = the trend itself). Default 0. The IPI's integrated
        squared derivative :math:`I[m^{(k)}]` uses ``v = k = p_trend + 1``.
    p : int, optional
        The local polynomial degree; must satisfy ``p >= v + 1`` with ``p - v`` **odd**. Default 3.
    mu : int, optional
        The kernel smoothness order — a key of :data:`KERNELS` (0 uniform, 1 Epanechnikov, 2
        bisquare, 3 triweight). Default 1.
    bandwidth : float, optional
        The relative bandwidth :math:`b \in (0, 0.5)`; the bandwidth in points is
        :math:`m = \lfloor n b \rfloor`. Default 0.15.
    boundary : {"fixed", "knn"}, optional
        The boundary rule: ``"fixed"`` (smoots ``bb=0``) truncates the window; ``"knn"`` (smoots
        ``bb=1``, the default) keeps :math:`2m+1` nearest points.

    Returns
    -------
    ndarray, shape (n,)
        The local-polynomial estimate :math:`\hat m^{(v)}(x_t)`.

    Raises
    ------
    ValueError
        If ``v`` is negative, ``p`` does not satisfy ``p >= v + 1`` with ``p - v`` odd, ``mu`` is
        not a supported kernel order, ``bandwidth`` is not in ``(0, 0.5)``, or the series is too
        short for the window.
    """
    values = np.asarray(y, dtype=np.float64)
    n = values.size
    if v < 0:
        raise ValueError(f"v must be a non-negative integer, got {v}")
    if p < v + 1 or (p - v) % 2 == 0:
        raise ValueError(f"require p >= v+1 and (p - v) odd, got p={p}, v={v}")
    if mu not in KERNELS:
        raise ValueError(f"mu must be one of {sorted(KERNELS)}, got {mu}")
    if not 0.0 < bandwidth < 0.5:
        raise ValueError(f"bandwidth must be in (0, 0.5), got {bandwidth}")
    if boundary not in ("fixed", "knn"):
        raise ValueError(f"boundary must be 'fixed' or 'knn', got {boundary!r}")
    m = int(np.floor(n * bandwidth))
    if n < 2 * m + 1:
        raise ValueError(f"series too short: n={n} < 2m+1={2 * m + 1} for bandwidth {bandwidth}")
    if m < p:
        raise ValueError(f"bandwidth too small: window half-width m={m} < p={p}")

    derivative_factor = factorial(v) * (n**v)  # v! * n^v: coefficient -> v-th x-derivative
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        if boundary == "fixed":
            lo, hi = max(0, t - m), min(n - 1, t + m)
        elif t < m:
            lo, hi = 0, 2 * m
        elif t > n - 1 - m:
            lo, hi = n - 1 - 2 * m, n - 1
        else:
            lo, hi = t - m, t + m
        offset = np.arange(lo - t, hi - t + 1, dtype=np.float64)
        scale = max(t - lo, hi - t) + 1
        weights = _kernel_weights(offset / scale, mu)
        design = np.vander(offset, p + 1, increasing=True)
        weighted = design.T * weights
        out[t] = np.linalg.solve(weighted @ design, weighted @ values[lo : hi + 1])[v]
    return np.asarray(out * derivative_factor, dtype=np.float64)


def integrated_squared_derivative(
    y: FloatArray,
    *,
    k: int,
    p: int,
    mu: int = 1,
    bandwidth: float = 0.15,
    boundary: Literal["fixed", "knn"] = "knn",
    cb: float = 0.05,
    db: float = 0.95,
) -> float:
    r"""The integrated squared ``k``-th derivative :math:`I[m^{(k)}]` (the AMISE numerator).

    The AMISE numerator of the IPI: estimate :math:`m^{(k)}` by :func:`local_poly` (``v = k``) at
    the given bandwidth, then **trapezoidally** integrate its square over the interior
    :math:`x \in [c_b, d_b]` (the equidistant grid points there), matching Feng-Gries-Fritz (2020).

    .. note::

        In the full IPI this is evaluated at an *inflated pilot bandwidth* :math:`h_d = h^a` whose
        value comes from the (Feng-Heiler-2009-tabulated) kernel constants — so a direct match to a
        fixture's stored ``I2`` requires that pilot bandwidth (the AMISE iteration, deferred). This
        function supplies the integrand/quadrature mechanism only.

    Parameters
    ----------
    y : ndarray, shape (n,)
        The equidistant series.
    k : int
        The derivative order (``k = p_trend + 1``).
    p : int
        The local polynomial degree for the derivative estimate (``p - k`` odd, ``p >= k + 1``).
    mu : int, optional
        The kernel smoothness order (default 1).
    bandwidth : float, optional
        The (pilot) bandwidth at which :math:`m^{(k)}` is estimated (default 0.15).
    boundary : {"fixed", "knn"}, optional
        The boundary rule (default ``"knn"``).
    cb, db : float, optional
        The interior integration limits on the rescaled time axis (default ``0.05``, ``0.95``).

    Returns
    -------
    float
        The integrated squared derivative over :math:`[c_b, d_b]`.

    Raises
    ------
    ValueError
        If ``cb``/``db`` are not ``0 <= cb < db <= 1``.
    """
    if not 0.0 <= cb < db <= 1.0:
        raise ValueError(f"require 0 <= cb < db <= 1, got cb={cb}, db={db}")
    mk = local_poly(y, v=k, p=p, mu=mu, bandwidth=bandwidth, boundary=boundary)
    n = mk.size
    x = (np.arange(1, n + 1, dtype=np.float64)) / n
    mask = (x >= cb) & (x <= db)
    xs = x[mask]
    sq = mk[mask] ** 2
    # Trapezoidal rule, written explicitly (numpy-version-agnostic: avoids trapz/trapezoid naming).
    return float(np.sum(0.5 * (sq[1:] + sq[:-1]) * np.diff(xs)))


def bartlett_variance_factor(residuals: FloatArray, *, window: int) -> float:
    r"""The short-memory variance factor :math:`c_f` by a Bartlett lag-window, given window ``M``.

    .. math::

        \hat c_f = \hat\gamma_0 + 2\sum_{l=1}^{M}\left(1 - \frac{l}{M+1}\right)\hat\gamma_l,

    with :math:`\hat\gamma_l` the **biased** (divide-by-:math:`n`) sample autocovariances of the
    (detrended) ``residuals`` and :math:`M` the lag-window width. Validated against ``smoots``'
    ``Mcf = "NP"`` output: recomputing the residuals from the series at the selected bandwidth and
    applying this sum reproduces ``cf0`` exactly.

    .. note::

        **Paper-vs-implementation divergence (documented).** Feng-Gries-Fritz (2020) write the
        Bartlett weights as :math:`1 - |l|/(M + 0.5)`; smoots' implementation uses
        :math:`1 - |l|/(M + 1)`. Validating against OUTPUT (CLAUDE.md §12), this uses ``M + 1``.
        **The automatic selection of the window ``M`` (``L0.opt``) is NOT implemented** — it is the
        Bühlmann (1996) nested spectral-density IPI, whose defining equations are not reproduced in
        Feng-Gries-Fritz (2020) and whose source (Bühlmann 1996) is unavailable; ``M`` must
        therefore be supplied (smoots permits manual control). See ``docs/fegarch-spec-notes.md``.

    Parameters
    ----------
    residuals : ndarray, shape (n,)
        The detrended residuals.
    window : int
        The Bartlett lag-window width :math:`M \ge 0` (the smoots ``L0.opt``, supplied).

    Returns
    -------
    float
        The variance factor :math:`\hat c_f`.

    Raises
    ------
    ValueError
        If ``window`` is negative or not smaller than the series length.
    """
    r = np.asarray(residuals, dtype=np.float64)
    n = r.size
    if window < 0 or window >= n:
        raise ValueError(f"window must satisfy 0 <= window < n={n}, got {window}")
    r = r - r.mean()
    gamma = np.array([float(np.sum(r[: n - lag] * r[lag:]) / n) for lag in range(window + 1)])
    weights = 1.0 - np.arange(1, window + 1) / (window + 1)
    return float(gamma[0] + 2.0 * np.sum(weights * gamma[1:]))


@dataclass(frozen=True)
class SemiparametricFit:
    r"""A fixed-bandwidth semiparametric EGF fit (nonparametric scale + zero-mean parametric EGF).

    The total volatility decomposes as :math:`\sigma_t = \hat s(x_t)\,\tilde\sigma_t` — a smooth
    deterministic scale :math:`\hat s` and the parametric conditional SD :math:`\tilde\sigma` of the
    de-scaled returns :math:`\hat r_t = (r_t - \bar y)/\hat s_t`.

    Attributes
    ----------
    mu : float
        The mean :math:`\hat\mu = \bar y` (the sample mean — the semiparametric mean is not fitted).
    c_t : float
        The empirical retransform correction :math:`\hat C_T = -\ln[n^{-1}\sum_t e^{\hat u_t}]`.
    scale : ndarray, shape (n,)
        The nonparametric scale :math:`\hat s(x_t)`.
    parametric_vol : ndarray, shape (n,)
        The parametric conditional SD :math:`\tilde\sigma_t` of :math:`\hat r_t` (``lambda_hat``).
    total_vol : ndarray, shape (n,)
        The total volatility :math:`\hat\Omega_t = \hat s_t\,\tilde\sigma_t` (fEGarch's ``sigt``).
    pars : dict of str to float
        The zero-mean parametric EGF parameters (``omega_sig, phi1, kappa, gamma`` [, ``d``]).
    loglikelihood : float
        The parametric-stage maximized log-likelihood (on :math:`\hat r_t`).
    model : str
        The parametric EGF model (``"egarch"`` or ``"fiegarch"``).
    bwidth : float
        The fixed relative bandwidth used for the scale.
    """

    mu: float
    c_t: float
    scale: FloatArray
    parametric_vol: FloatArray
    total_vol: FloatArray
    pars: dict[str, float]
    loglikelihood: float
    model: str
    bwidth: float


def semiparametric_scale(
    returns: FloatArray,
    *,
    poly_order: int = 1,
    kernel_order: int = 1,
    boundary_method: Literal["extend", "shorten"] = "extend",
    bwidth: float,
) -> tuple[float, float, FloatArray, FloatArray]:
    r"""The semiparametric scale :math:`\hat s(x_t)` at a fixed bandwidth (WP171 §2.2.1 steps 1--5).

    The machine-exact, model-independent scale core: ``(1)`` :math:`\bar y = \mathrm{mean}(y)`,
    :math:`\tilde w_t = \ln[(y_t-\bar y)^2]`; ``(2)`` :math:`\hat m(x_t)` = :func:`local_poly` of
    :math:`\tilde w` at the fixed ``bwidth``; ``(3)`` the **empirical** retransform correction
    :math:`\hat C_T = -\ln[n^{-1}\sum_t e^{\hat u_t}]`, :math:`\hat u_t = \tilde w_t - \hat m_t`
    (the finite-sample-unbiasing correction forcing :math:`\operatorname{E}[e^{\hat u}] = 1` —
    **not** the theoretical :math:`\operatorname{E}[\ln\zeta^2]`); ``(4)``
    :math:`\hat s(x_t) = \exp\{(\hat m_t - \hat C_T)/2\}`; ``(5)``
    :math:`\hat r_t = (y_t-\bar y)/\hat s_t`. Route-independent at a fixed bandwidth; reproduces
    fEGarch's ``scale_fun`` to machine precision.

    Parameters
    ----------
    returns : ndarray, shape (n,)
        The return series.
    poly_order : int, optional
        The local polynomial order (``locpol_spec`` ``poly_order``; 1 or 3). Default 1.
    kernel_order : int, optional
        The kernel smoothness order (``locpol_spec`` ``kernel_order`` / :data:`KERNELS`). Default 1.
    boundary_method : {"extend", "shorten"}, optional
        ``locpol_spec`` boundary method: ``"extend"`` (= gsmooth k-NN) or ``"shorten"`` (= fixed).
    bwidth : float
        The fixed relative bandwidth :math:`\in (0, 0.5)` (keyword-only, required).

    Returns
    -------
    tuple
        ``(mu, c_t, scale, standardized)`` — :math:`\bar y`, :math:`\hat C_T`, :math:`\hat s(x_t)`
        and the de-scaled returns :math:`\hat r_t`.

    Raises
    ------
    ValueError
        If ``boundary_method`` is not ``"extend"``/``"shorten"``.
    """
    if boundary_method not in _BOUNDARY_MAP:
        raise ValueError(f"boundary_method must be 'extend' or 'shorten', got {boundary_method!r}")
    y = np.asarray(returns, dtype=np.float64)
    mu = float(y.mean())
    demeaned = y - mu
    log_sq = np.log(demeaned**2)
    m_hat = local_poly(
        log_sq,
        v=0,
        p=poly_order,
        mu=kernel_order,
        bandwidth=bwidth,
        boundary=_BOUNDARY_MAP[boundary_method],
    )
    c_t = -float(np.log(np.mean(np.exp(log_sq - m_hat))))
    scale = np.exp((m_hat - c_t) / 2.0)
    standardized = demeaned / scale
    return mu, c_t, np.asarray(scale, dtype=np.float64), np.asarray(standardized, dtype=np.float64)


def semiparametric_fit(
    returns: FloatArray,
    *,
    model: str = "egarch",
    cond_dist: str = "norm",
    poly_order: int = 1,
    kernel_order: int = 1,
    boundary_method: Literal["extend", "shorten"] = "extend",
    bwidth: float | None,
) -> SemiparametricFit:
    r"""Fixed-bandwidth semiparametric fit for **any** fEGarch inner model (WP171 §2.2.1, 6 steps).

    Estimates the nonparametric scale (:func:`semiparametric_scale`, steps 1--5, machine-exact and
    **model-/distribution-agnostic** — the empirical :math:`\hat C_T` depends only on the returns
    and the local-poly trend), then fits a **zero-mean** inner model to the de-scaled returns
    :math:`\hat r_t = (y_t-\bar y)/\hat s_t` (step 6) by dispatching to that model's validated
    ``fit_*(..., include_mean=False)``, returning the total volatility
    :math:`\hat\Omega = \hat s\,\tilde\sigma` (fEGarch's ``sigt``). The inner stage reuses the exact
    parametric fitter with the mean fixed at 0 (the mean is already :math:`\bar y`), so it inherits
    that model's ALD ``P``-profiling, per-iteration centering and restart logic — the semiparametric
    fit is **scale (machine-exact) ⊕ the already-validated parametric fit**. Any of the 16 inner
    models (:data:`_INNER_FITTERS`) is accepted.

    .. note::

        **Automatic bandwidth is not implemented** (``bwidth=None`` raises): it needs the Bühlmann
        (1996) + Feng-Heiler (2009) machinery, unavailable in ``literature/``. Supply a fixed
        ``bwidth`` in ``(0, 0.5)``. The supported conditional-distribution surface per model
        inherits the parametric fits' finding-3 drops (e.g. the Type-I EGF family drops the
        continuous-``df`` ``std``/``sstd``); on the near-normal de-scaled residuals the
        ``df`` is even more weakly identified, so the variance-recursion family's ``std``/``sstd``
        fits can climb the flat ``df`` ridge past fEGarch's finite-``df`` stall — an optimizer
        divergence, not a defect (see ``docs/fegarch-spec-notes.md``). Forecasting-with-scale and
        dual-mean composition are deferred follow-ons.

    Parameters
    ----------
    returns : ndarray, shape (n,)
        The return series.
    model : str, optional
        The inner model fit to the de-scaled returns — a key of :data:`_INNER_FITTERS` (one of the
        16 fEGarch models). Default ``"egarch"``.
    cond_dist : str, optional
        The conditional distribution (any the inner model's fitter accepts). Default ``"norm"``.
    poly_order, kernel_order : int, optional
        The local-polynomial order and kernel smoothness for the scale (default 1, 1).
    boundary_method : {"extend", "shorten"}, optional
        The scale boundary method (default ``"extend"``).
    bwidth : float or None
        The fixed relative bandwidth in ``(0, 0.5)``; ``None`` raises ``NotImplementedError``.

    Returns
    -------
    SemiparametricFit
        The scale, parametric and total volatilities, and the zero-mean inner-model parameters.

    Raises
    ------
    NotImplementedError
        If ``bwidth`` is ``None`` (automatic bandwidth selection is out of scope).
    ValueError
        If ``model`` is not one of the 16 supported inner models.
    """
    if bwidth is None:
        raise NotImplementedError(
            "automatic bandwidth selection is not implemented (needs Bühlmann 1996 + "
            "Feng-Heiler 2009, unavailable); supply a fixed bwidth in (0, 0.5)"
        )
    if model not in _INNER_FITTERS:
        raise ValueError(f"model must be one of {sorted(_INNER_FITTERS)}, got {model!r}")

    mu, c_t, scale, r_hat = semiparametric_scale(
        returns,
        poly_order=poly_order,
        kernel_order=kernel_order,
        boundary_method=boundary_method,
        bwidth=bwidth,
    )
    # Step 6: the zero-mean inner fit on the de-scaled returns (the exact validated parametric
    # fitter, mean fixed at 0). exp overflow can occur in bad optimizer regions of the EGF models.
    with np.errstate(over="ignore", invalid="ignore"):
        inner = _INNER_FITTERS[model](r_hat, cond_dist=cond_dist, include_mean=False)
    parametric_vol = np.asarray(
        inner.conditional_volatility, dtype=np.float64
    )  # lambda_hat on rhat
    total_vol = scale * parametric_vol
    return SemiparametricFit(
        mu=mu,
        c_t=c_t,
        scale=scale,
        parametric_vol=parametric_vol,
        total_vol=np.asarray(total_vol, dtype=np.float64),
        pars=dict(inner.params),
        loglikelihood=float(inner.loglikelihood),
        model=model,
        bwidth=float(bwidth),
    )
