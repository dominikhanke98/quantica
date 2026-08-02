r"""Conditional innovation distributions for the fEGarch clean-room port (Phase 0).

.. note::

    **Clean-room (CLAUDE.md §12).** This is an *independent clean-room reimplementation of the
    models described in the fEGarch reference manual and its cited papers* — implemented purely from
    the published mathematics, never from the fEGarch source. It is validated against committed
    fEGarch *output* fixtures (a later step), not by inspecting its code.

The eight conditional distributions the fEGarch family uses for the standardized innovation
:math:`z_t = \varepsilon_t / \sigma_t`, each **standardized to mean 0 and variance 1** (the QMLE
convention, so the conditional variance carries the whole scale):

* symmetric bases — ``norm`` (standard normal), ``std`` (standardized Student-:math:`t`),
  ``ged`` (generalized error distribution, Nelson-1991 standardization), ``ald`` (average Laplace);
* their **Fernández-Steel (1998) skewed** counterparts — ``snorm``, ``sstd``, ``sged``, ``sald`` —
  a skewness parameter :math:`\xi` splits the density and the result is re-standardized to mean 0 /
  variance 1 (Lambert-Laurent 2001), so :math:`\xi = 1` recovers the symmetric base exactly.

Each distribution exposes ``logpdf`` / ``pdf`` / ``cdf`` / ``ppf`` and a seeded ``sample`` (drawn by
inverse-CDF so it is deterministic and consistent with ``ppf``).

Two parameterizations are **flagged for reconciliation** against the papers + fEGarch fixtures once
those exist (documented defaults are implemented now; the API will not change when the exact
constant is locked):

* the exact **average-Laplace (``ald``) form** — implemented here as the standardized symmetric
  Laplace as a documented default (see :class:`AverageLaplace`);
* the exact **Fernández-Steel normalization / skew parameterization** — implemented here via the
  standard Lambert-Laurent mean-0/variance-1 constants (see :class:`FernandezSteelSkew`).

References
----------
Nelson, D. B. (1991). "Conditional Heteroskedasticity in Asset Returns: A New Approach."
*Econometrica* 59(2) — the standardized GED in the GARCH context.
Fernández, C. & Steel, M. F. J. (1998). "On Bayesian Modeling of Fat Tails and Skewness." *JASA*
93(441) — the density-splitting skew mechanism.
Lambert, P. & Laurent, S. (2001). "Modelling financial time series using GARCH-type models with a
skewed Student distribution for the innovations." — the mean-0/variance-1 standardization used here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy import special, stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantica.core.types import FloatArray

__all__ = [
    "DISTRIBUTIONS",
    "AverageLaplace",
    "ConditionalDistribution",
    "FernandezSteelSkew",
    "GeneralizedError",
    "Normal",
    "StudentT",
    "get_distribution",
]


class ConditionalDistribution(ABC):
    """A standardized (mean-0, variance-1) innovation distribution for QMLE.

    Every distribution has a fixed tuple of *shape* parameters (``param_names``), passed to the
    methods as a sequence aligned with those names (empty for the parameter-free normal). Methods
    operate elementwise on ``z``/``p`` arrays.

    Attributes
    ----------
    name : str
        The fEGarch short code (``"norm"``, ``"std"``, ...).
    param_names : tuple of str
        The shape-parameter names, in order.
    param_bounds : tuple of (float, float)
        Optimization bounds for each shape parameter, aligned with ``param_names``.
    param_start : tuple of float
        Sensible starting values for each shape parameter.
    """

    name: str = ""
    param_names: tuple[str, ...] = ()
    param_bounds: tuple[tuple[float, float], ...] = ()
    param_start: tuple[float, ...] = ()

    def _params(self, params: Sequence[float] | None) -> tuple[float, ...]:
        """Return ``params`` as a validated float tuple, defaulting to ``param_start``."""
        values = tuple(self.param_start) if params is None else tuple(float(p) for p in params)
        if len(values) != len(self.param_names):
            raise ValueError(
                f"{self.name} expects {len(self.param_names)} params, got {len(values)}"
            )
        return values

    @abstractmethod
    def logpdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Log density of the standardized innovation at ``z``."""

    def pdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Density of the standardized innovation at ``z``."""
        return np.asarray(np.exp(self.logpdf(z, params)), dtype=np.float64)

    @abstractmethod
    def cdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Cumulative distribution function at ``z``."""

    @abstractmethod
    def ppf(self, p: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Quantile function (inverse CDF) at probability ``p``."""

    def sample(
        self, size: int, rng: np.random.Generator, params: Sequence[float] | None = None
    ) -> FloatArray:
        """Draw ``size`` seeded samples by inverse-CDF (consistent with :meth:`ppf`)."""
        return self.ppf(rng.uniform(size=size), params)


# --------------------------------------------------------------------------- #
# Symmetric bases
# --------------------------------------------------------------------------- #


class Normal(ConditionalDistribution):
    """The standard normal innovation (``norm``): mean 0, variance 1, no shape parameters."""

    name = "norm"

    def logpdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Log density :math:`-\\tfrac12\\ln(2\\pi) - \\tfrac12 z^2`."""
        self._params(params)
        z = np.asarray(z, dtype=np.float64)
        return np.asarray(-0.5 * np.log(2.0 * np.pi) - 0.5 * z * z, dtype=np.float64)

    def cdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standard normal CDF."""
        self._params(params)
        return np.asarray(special.ndtr(np.asarray(z, dtype=np.float64)), dtype=np.float64)

    def ppf(self, p: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standard normal quantile function."""
        self._params(params)
        return np.asarray(special.ndtri(np.asarray(p, dtype=np.float64)), dtype=np.float64)

    def abs_moment(self, params: Sequence[float] | None = None) -> float:
        """First absolute moment :math:`E|z| = \\sqrt{2/\\pi}` (needed by the skew wrapper)."""
        self._params(params)
        return float(np.sqrt(2.0 / np.pi))


class StudentT(ConditionalDistribution):
    r"""The standardized Student-:math:`t` innovation (``std``), shape :math:`\nu > 2`.

    The scale is chosen so the variance is 1: if :math:`T \sim t_\nu` then
    :math:`z = T\sqrt{(\nu-2)/\nu}`. The density is
    :math:`\frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{(\nu-2)\pi}\,\Gamma(\frac{\nu}{2})}
    (1 + \frac{z^2}{\nu-2})^{-(\nu+1)/2}`.
    """

    name = "std"
    param_names = ("nu",)
    param_bounds = ((2.05, 100.0),)
    param_start = (8.0,)

    def logpdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized Student-``t`` log density."""
        (nu,) = self._params(params)
        z = np.asarray(z, dtype=np.float64)
        const = special.gammaln((nu + 1.0) / 2.0) - special.gammaln(nu / 2.0)
        const -= 0.5 * np.log((nu - 2.0) * np.pi)
        return np.asarray(const - (nu + 1.0) / 2.0 * np.log1p(z * z / (nu - 2.0)), dtype=np.float64)

    def cdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized Student-``t`` CDF (via the scaled ordinary ``t``)."""
        (nu,) = self._params(params)
        z = np.asarray(z, dtype=np.float64)
        return np.asarray(stats.t.cdf(z * np.sqrt(nu / (nu - 2.0)), nu), dtype=np.float64)

    def ppf(self, p: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized Student-``t`` quantile function."""
        (nu,) = self._params(params)
        p = np.asarray(p, dtype=np.float64)
        return np.asarray(stats.t.ppf(p, nu) * np.sqrt((nu - 2.0) / nu), dtype=np.float64)

    def abs_moment(self, params: Sequence[float] | None = None) -> float:
        """First absolute moment of the standardized ``t``."""
        (nu,) = self._params(params)
        log_val = special.gammaln((nu + 1.0) / 2.0) - special.gammaln(nu / 2.0)
        return float(2.0 * np.sqrt(nu - 2.0) * np.exp(log_val) / (np.sqrt(np.pi) * (nu - 1.0)))


class GeneralizedError(ConditionalDistribution):
    r"""The standardized generalized error distribution (``ged``), shape :math:`\nu > 0`.

    Nelson's (1991) GED with the unit-variance standardization: :math:`\nu = 2` is the standard
    normal, :math:`\nu < 2` is heavier-tailed, :math:`\nu > 2` thinner. Implemented via
    :func:`scipy.stats.gennorm` (whose shape :math:`\beta` equals :math:`\nu`) with the scale
    :math:`\sqrt{\Gamma(1/\nu)/\Gamma(3/\nu)}` that makes the variance 1.
    """

    name = "ged"
    param_names = ("nu",)
    param_bounds = ((0.5, 20.0),)
    param_start = (1.5,)

    @staticmethod
    def _scale(nu: float) -> float:
        """Unit-variance scale :math:`\\sqrt{\\Gamma(1/\\nu)/\\Gamma(3/\\nu)}` for ``gennorm``."""
        return float(np.sqrt(np.exp(special.gammaln(1.0 / nu) - special.gammaln(3.0 / nu))))

    def logpdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized GED log density."""
        (nu,) = self._params(params)
        z = np.asarray(z, dtype=np.float64)
        return np.asarray(stats.gennorm.logpdf(z, beta=nu, scale=self._scale(nu)), dtype=np.float64)

    def cdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized GED CDF."""
        (nu,) = self._params(params)
        z = np.asarray(z, dtype=np.float64)
        return np.asarray(stats.gennorm.cdf(z, beta=nu, scale=self._scale(nu)), dtype=np.float64)

    def ppf(self, p: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized GED quantile function."""
        (nu,) = self._params(params)
        p = np.asarray(p, dtype=np.float64)
        return np.asarray(stats.gennorm.ppf(p, beta=nu, scale=self._scale(nu)), dtype=np.float64)

    def abs_moment(self, params: Sequence[float] | None = None) -> float:
        """First absolute moment :math:`\\Gamma(2/\\nu)/\\sqrt{\\Gamma(1/\\nu)\\Gamma(3/\\nu)}`."""
        (nu,) = self._params(params)
        log_val = special.gammaln(2.0 / nu) - 0.5 * (
            special.gammaln(1.0 / nu) + special.gammaln(3.0 / nu)
        )
        return float(np.exp(log_val))


class AverageLaplace(ConditionalDistribution):
    r"""The average-Laplace innovation (``ald``) — **RECONCILE default: standardized Laplace**.

    The exact *average Laplace* form used by fEGarch is not yet pinned from a citable paper, so this
    implements the standardized symmetric **Laplace** (double-exponential, scale :math:`1/\sqrt2`
    for unit variance) as a well-defined, documented default. The interface (parameter-free,
    standardized) is designed so swapping in the exact form later leaves the API unchanged.

    ``# RECONCILE``: confirm the exact ``ald`` density against the fEGarch reference manual / cited
    paper and the committed fEGarch fixture, then replace the body without changing this signature.
    """

    name = "ald"
    _SCALE = 1.0 / np.sqrt(2.0)  # unit-variance Laplace scale

    def logpdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized Laplace log density (RECONCILE default)."""
        self._params(params)
        z = np.asarray(z, dtype=np.float64)
        return np.asarray(stats.laplace.logpdf(z, scale=self._SCALE), dtype=np.float64)

    def cdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized Laplace CDF (RECONCILE default)."""
        self._params(params)
        z = np.asarray(z, dtype=np.float64)
        return np.asarray(stats.laplace.cdf(z, scale=self._SCALE), dtype=np.float64)

    def ppf(self, p: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized Laplace quantile function (RECONCILE default)."""
        self._params(params)
        p = np.asarray(p, dtype=np.float64)
        return np.asarray(stats.laplace.ppf(p, scale=self._SCALE), dtype=np.float64)

    def abs_moment(self, params: Sequence[float] | None = None) -> float:
        """First absolute moment :math:`E|z| = 1/\\sqrt2` (RECONCILE default)."""
        self._params(params)
        return float(self._SCALE)


# --------------------------------------------------------------------------- #
# Fernández-Steel skew wrapper
# --------------------------------------------------------------------------- #

#: Symmetric bases that expose the first absolute moment needed to standardize the skew.
_SymmetricBase = Normal | StudentT | GeneralizedError | AverageLaplace


class FernandezSteelSkew(ConditionalDistribution):
    r"""Fernández-Steel (1998) skew of a symmetric base, re-standardized to mean 0 / variance 1.

    Given a standardized symmetric base density :math:`f` and skew :math:`\xi > 0`, the
    unstandardized skewed density is
    :math:`h(x) = \frac{2}{\xi+1/\xi} f(x\,\xi^{-\operatorname{sign} x})`. Its mean
    :math:`\mu = M_1(\xi - 1/\xi)` and variance
    :math:`\sigma^2 = (1-M_1^2)(\xi^2+\xi^{-2}) + 2M_1^2 - 1` (with :math:`M_1 = E|z_{\text{base}}|`
    the base's first absolute moment) are removed so the innovation is again mean 0 / variance 1
    (Lambert-Laurent 2001). At :math:`\xi = 1` this reduces exactly to the base.

    ``# RECONCILE``: confirm this is the exact skew parameterization and normalization fEGarch uses
    (vs. an alternative :math:`\xi`/skew constant) against the paper and the fEGarch fixture; the
    single ``xi`` parameter and the mean-0/variance-1 convention are designed to stay fixed.
    """

    def __init__(self, base: _SymmetricBase) -> None:
        self._base = base
        self.name = "s" + base.name
        self.param_names = (*base.param_names, "xi")
        self.param_bounds = (*base.param_bounds, (0.1, 10.0))
        self.param_start = (*base.param_start, 1.0)

    def _split(self, params: Sequence[float] | None) -> tuple[tuple[float, ...], float]:
        """Split the parameter vector into (base shape params, skew xi)."""
        values = self._params(params)
        return values[:-1], values[-1]

    def _mu_sigma(self, base_params: tuple[float, ...], xi: float) -> tuple[float, float]:
        """The standardizing mean and standard deviation of the skewed base."""
        m1 = self._base.abs_moment(base_params)
        mu = m1 * (xi - 1.0 / xi)
        sigma2 = (1.0 - m1 * m1) * (xi * xi + 1.0 / (xi * xi)) + 2.0 * m1 * m1 - 1.0
        return mu, float(np.sqrt(sigma2))

    def logpdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized skewed log density."""
        base_params, xi = self._split(params)
        mu, sigma = self._mu_sigma(base_params, xi)
        z_star = sigma * np.asarray(z, dtype=np.float64) + mu
        arg = z_star * np.power(xi, -np.sign(z_star))
        log_norm = np.log(2.0 * sigma) - np.log(xi + 1.0 / xi)
        return np.asarray(log_norm + self._base.logpdf(arg, base_params), dtype=np.float64)

    def cdf(self, z: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized skewed CDF."""
        base_params, xi = self._split(params)
        mu, sigma = self._mu_sigma(base_params, xi)
        a = sigma * np.asarray(z, dtype=np.float64) + mu
        xi2 = xi * xi
        neg = 2.0 / (xi2 + 1.0) * self._base.cdf(a * xi, base_params)
        pos = 1.0 / (xi2 + 1.0) + 2.0 * xi2 / (xi2 + 1.0) * (
            self._base.cdf(a / xi, base_params) - 0.5
        )
        return np.asarray(np.where(a < 0.0, neg, pos), dtype=np.float64)

    def ppf(self, p: FloatArray, params: Sequence[float] | None = None) -> FloatArray:
        """Standardized skewed quantile function."""
        base_params, xi = self._split(params)
        mu, sigma = self._mu_sigma(base_params, xi)
        p = np.asarray(p, dtype=np.float64)
        xi2 = xi * xi
        p0 = 1.0 / (xi2 + 1.0)  # CDF at a = 0
        lower = self._base.ppf(p * (xi2 + 1.0) / 2.0, base_params) / xi
        upper = xi * self._base.ppf(0.5 + (p * (xi2 + 1.0) - 1.0) / (2.0 * xi2), base_params)
        a = np.where(p <= p0, lower, upper)
        return np.asarray((a - mu) / sigma, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: The eight fEGarch conditional distributions, keyed by their fEGarch short code.
DISTRIBUTIONS: dict[str, ConditionalDistribution] = {
    "norm": Normal(),
    "std": StudentT(),
    "ged": GeneralizedError(),
    "ald": AverageLaplace(),
    "snorm": FernandezSteelSkew(Normal()),
    "sstd": FernandezSteelSkew(StudentT()),
    "sged": FernandezSteelSkew(GeneralizedError()),
    "sald": FernandezSteelSkew(AverageLaplace()),
}


def get_distribution(name: str) -> ConditionalDistribution:
    """Look up a conditional distribution by its fEGarch short code.

    Parameters
    ----------
    name : str
        One of ``norm``, ``std``, ``ged``, ``ald``, ``snorm``, ``sstd``, ``sged``, ``sald``.

    Returns
    -------
    ConditionalDistribution
        The (standardized) distribution instance.

    Raises
    ------
    KeyError
        If ``name`` is not one of the eight supported codes.
    """
    try:
        return DISTRIBUTIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown distribution {name!r}; expected one of {sorted(DISTRIBUTIONS)}"
        ) from None
