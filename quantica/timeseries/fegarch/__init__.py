r"""fEGarch clean-room port — an independent reimplementation of the fEGarch model family.

.. note::

    **Clean-room, non-negotiable (CLAUDE.md §12).** This package is an **independent clean-room
    reimplementation of the models described in the `fEGarch` reference manual and its cited
    papers** — implemented purely from the published mathematics, **never** from the `fEGarch`
    R/C++ source (which is never read, opened, or translated). Correctness is validated against
    **committed `fEGarch` output fixtures** (a pending step), not by inspecting their code.
    `fEGarch` and the underlying papers are credited as the *specification source*; this is not a
    port of their code and implies no endorsement. `quantica` stays MIT because the work is
    original, from specifications.

**Phase 0 (this step) — the foundations** every later model reuses:

* **Conditional distributions** (:mod:`~quantica.timeseries.fegarch.distributions`) — the eight
  standardized (mean-0, variance-1) innovation distributions ``norm`` / ``std`` / ``ged`` / ``ald``
  and their Fernández-Steel skewed variants, each with ``pdf`` / ``cdf`` / ``ppf`` / seeded sampler.
* **QMLE engine** (:mod:`~quantica.timeseries.fegarch.qmle`) — a generic quasi-maximum-likelihood
  estimator that fits any conditional-variance recursion under any of those distributions, with
  documented pre-sample conditioning and Hessian-based standard errors.

The models (short-memory, EGARCH family, the fractional-differencing engine, long-memory,
dual mean, forecasting/risk) arrive in later phases — see ``docs/fegarch-port-roadmap.md``.
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
    "GeneralizedError",
    "Normal",
    "QMLEResult",
    "StudentT",
    "VarianceRecursion",
    "get_distribution",
    "initial_variance",
    "quasi_max_likelihood",
]
