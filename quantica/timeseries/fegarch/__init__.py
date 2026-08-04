r"""fEGarch clean-room port — an independent reimplementation of the fEGarch model family.

.. note::

    **Clean-room, non-negotiable (CLAUDE.md §12).** This package is an **independent clean-room
    reimplementation of the models described in the `fEGarch` reference manual and its cited
    papers** — implemented purely from the published mathematics, **never** from the `fEGarch`
    R/C++ source (which is never read, opened, or translated). Correctness is validated against
    **committed `fEGarch` output fixtures** (``tests/fixtures/fegarch/``), not by inspecting their
    code. `fEGarch` and the underlying papers are credited as the *specification source*; this is
    not a port of their code and implies no endorsement. `quantica` stays MIT because the work is
    original, from specifications.

**Phase 0 — the foundations** every model reuses:

* **Conditional distributions** (:mod:`~quantica.timeseries.fegarch.distributions`) — the eight
  standardized (mean-0, variance-1) innovation distributions ``norm`` / ``std`` / ``ged`` / ``ald``
  and their Fernández-Steel skewed variants, each with ``pdf`` / ``cdf`` / ``ppf`` / seeded sampler.
* **QMLE engine** (:mod:`~quantica.timeseries.fegarch.qmle`) — a generic quasi-maximum-likelihood
  estimator that fits any conditional-variance recursion under any of those distributions, with a
  constant-mean option, documented pre-sample conditioning and Hessian-based standard errors.

**Phase 1 — short-memory models** (in progress):

* **GARCH(1,1)** (:mod:`~quantica.timeseries.fegarch.garch`) — the Bollerslev (1986) recursion as a
  variance recursion on the Phase-0 engine (:func:`~quantica.timeseries.fegarch.fit_garch`,
  :func:`~quantica.timeseries.fegarch.garch_sim`), reproducing fEGarch's fit (parameters,
  log-likelihood and conditional-SD series) on the committed fixture. GJR-GARCH / TGARCH / APARCH
  are the remaining Phase-1 models.

The EGARCH family, the fractional-differencing engine, the long-memory models, the dual mean and the
forecasting/risk tie-back arrive in later phases — see ``docs/fegarch-port-roadmap.md``.
"""

from __future__ import annotations

from quantica.timeseries.fegarch.distributions import (
    DISTRIBUTIONS,
    AverageLaplace,
    ConditionalDistribution,
    FernandezSteelSkew,
    GeneralizedError,
    Normal,
    StudentT,
    get_distribution,
)
from quantica.timeseries.fegarch.garch import GarchFit, fit_garch, garch_recursion, garch_sim
from quantica.timeseries.fegarch.qmle import (
    QMLEResult,
    VarianceRecursion,
    initial_variance,
    quasi_max_likelihood,
)

__all__ = [
    "DISTRIBUTIONS",
    "AverageLaplace",
    "ConditionalDistribution",
    "FernandezSteelSkew",
    "GarchFit",
    "GeneralizedError",
    "Normal",
    "QMLEResult",
    "StudentT",
    "VarianceRecursion",
    "fit_garch",
    "garch_recursion",
    "garch_sim",
    "get_distribution",
    "initial_variance",
    "quasi_max_likelihood",
]
