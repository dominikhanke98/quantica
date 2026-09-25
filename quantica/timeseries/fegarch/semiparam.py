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

References
----------
Feng, Y., Gries, T. & Fritz, M. (2020). "Data-Driven Local Polynomial for the Trend and its
Derivatives in Economic Time Series." *Journal of Nonparametric Statistics* 32(2).
Beran, J. & Feng, Y. (2002). "SEMIFAR models — a semiparametric approach to modelling trends,
long-range dependence and nonstationarity." *Computational Statistics & Data Analysis* 40(2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "KERNELS",
    "local_poly",
]

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
    p: int = 3,
    mu: int = 1,
    bandwidth: float = 0.15,
    boundary: Literal["fixed", "knn"] = "knn",
) -> FloatArray:
    r"""Local-polynomial trend estimate :math:`\hat m(x_t)` at a fixed bandwidth (``gsmooth`` core).

    Weighted-least-squares local-polynomial regression of ``y`` on the rescaled equidistant time
    grid, returning the fitted trend (:math:`v = 0`). See the module docstring for the estimator,
    kernels and boundary conventions; this reproduces ``smoots::gsmooth(y, v=0, p, mu, b, bb)``
    machine-exactly.

    Parameters
    ----------
    y : ndarray, shape (n,)
        The equidistant series to smooth (e.g. the log-squared demeaned returns for the scale).
    p : int, optional
        The local polynomial degree; ``p`` must be odd (``p - v`` odd with :math:`v = 0`). Default
        3.
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
        The local-polynomial trend estimate :math:`\hat m(x_t)`.

    Raises
    ------
    ValueError
        If ``p`` is not a positive odd integer, ``mu`` is not a supported kernel order,
        ``bandwidth`` is not in ``(0, 0.5)``, or the series is too short for the window (``n < 2m +
        1``).
    """
    values = np.asarray(y, dtype=np.float64)
    n = values.size
    if p < 1 or p % 2 == 0:
        raise ValueError(f"p must be a positive odd integer (p - v odd, v=0), got {p}")
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
        out[t] = np.linalg.solve(weighted @ design, weighted @ values[lo : hi + 1])[0]
    return out
