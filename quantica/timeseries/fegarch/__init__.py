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

**Phase 1 — short-memory models** (complete):

* **GARCH(1,1)** (:mod:`~quantica.timeseries.fegarch.garch`) — the Bollerslev (1986) recursion as a
  variance recursion on the Phase-0 engine (:func:`~quantica.timeseries.fegarch.fit_garch`,
  :func:`~quantica.timeseries.fegarch.garch_sim`), reproducing fEGarch's fit (parameters,
  log-likelihood and conditional-SD series) on the committed fixture.
* **GJR-GARCH / TGARCH / APARCH** (:mod:`~quantica.timeseries.fegarch.asymmetric`) — the asymmetric
  short-memory models, reconciled against the fixtures as a **single APARCH power recursion**
  (Ding-Granger-Engle 1993) at power :math:`\delta = 2` (GJR; Glosten-Jagannathan-Runkle 1993),
  :math:`\delta = 1` (TGARCH; Zakoian 1994) and a free estimated :math:`\delta` (APARCH), via
  :func:`~quantica.timeseries.fegarch.fit_gjr`, :func:`~quantica.timeseries.fegarch.fit_tgarch`,
  :func:`~quantica.timeseries.fegarch.fit_aparch` and their simulators. This completes the Phase-1
  short-memory family (GARCH / GJR / TGARCH / APARCH).

**Phase 2 — EGARCH family** (complete for orders ``(1,1)`` / ``norm``):

* **EGARCH / MEGARCH / MLog-GARCH** (:mod:`~quantica.timeseries.fegarch.egarch`) — the **Type-I**
  EGF models, one generalized log-variance recursion :math:`\ln\sigma_t^2 = \omega + g(\eta_{t-1}) +
  \phi_1\ln\sigma_{t-1}^2` whose transformation :math:`g(\eta) = \kappa\{g_{\mathrm{asy}} -
  \operatorname{E}\} + \gamma\{g_{\mathrm{mag}} - \operatorname{E}\}` is parameterized by fixed
  construction constants (WP 2026-04 Eqs. 7-9): EGARCH ``(0,1,0,1)`` (Nelson 1991), MEGARCH and
  MLog-GARCH the modulus-log variants (Peitz et al. 2026), via
  :func:`~quantica.timeseries.fegarch.fit_egarch` / :func:`~quantica.timeseries.fegarch.fit_megarch`
  / :func:`~quantica.timeseries.fegarch.fit_mloggarch` and their simulators.
* **Log-GARCH(1,1)** (:mod:`~quantica.timeseries.fegarch.loggarch`) — the **Type-II** EGF model
  (Geweke 1986), an ARMA on the log-variance in the log-square innovation
  :math:`\xi = \ln\eta^2 - \operatorname{E}[\ln\eta^2]`, i.e. :math:`\ln\sigma_t^2 = \omega +
  \phi_1\ln\sigma_{t-1}^2 + (\psi_1 + \phi_1)\xi_{t-1}` (no asymmetry term), via
  :func:`~quantica.timeseries.fegarch.fit_loggarch` /
  :func:`~quantica.timeseries.fegarch.loggarch_sim`. This completes the four Phase-2 EGF models.

The fractional-differencing engine, the long-memory models, the dual mean and the forecasting/risk
tie-back arrive in later phases — see ``docs/fegarch-port-roadmap.md``.
"""

from __future__ import annotations

from quantica.timeseries.fegarch.asymmetric import (
    aparch_recursion,
    aparch_sim,
    fit_aparch,
    fit_gjr,
    fit_tgarch,
    gjr_recursion,
    gjr_sim,
    tgarch_recursion,
    tgarch_sim,
)
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
from quantica.timeseries.fegarch.egarch import (
    EGARCH_CONSTANTS,
    MEGARCH_CONSTANTS,
    MLOGGARCH_CONSTANTS,
    egarch_recursion,
    egarch_sim,
    fit_egarch,
    fit_megarch,
    fit_mloggarch,
    megarch_recursion,
    megarch_sim,
    mloggarch_recursion,
    mloggarch_sim,
    type1_news_impact,
)
from quantica.timeseries.fegarch.fiegarch import (
    fiegarch_recursion,
    fiegarch_sim,
    fit_fiegarch,
    theta_coefficients,
)
from quantica.timeseries.fegarch.fracdiff import fracdiff, fracdiff_coeffs
from quantica.timeseries.fegarch.garch import GarchFit, fit_garch, garch_recursion, garch_sim
from quantica.timeseries.fegarch.loggarch import fit_loggarch, loggarch_recursion, loggarch_sim
from quantica.timeseries.fegarch.qmle import (
    QMLEResult,
    VarianceRecursion,
    initial_variance,
    quasi_max_likelihood,
)

__all__ = [
    "DISTRIBUTIONS",
    "EGARCH_CONSTANTS",
    "MEGARCH_CONSTANTS",
    "MLOGGARCH_CONSTANTS",
    "AverageLaplace",
    "ConditionalDistribution",
    "FernandezSteelSkew",
    "GarchFit",
    "GeneralizedError",
    "Normal",
    "QMLEResult",
    "StudentT",
    "VarianceRecursion",
    "aparch_recursion",
    "aparch_sim",
    "egarch_recursion",
    "egarch_sim",
    "fiegarch_recursion",
    "fiegarch_sim",
    "fit_aparch",
    "fit_egarch",
    "fit_fiegarch",
    "fit_garch",
    "fit_gjr",
    "fit_loggarch",
    "fit_megarch",
    "fit_mloggarch",
    "fit_tgarch",
    "fracdiff",
    "fracdiff_coeffs",
    "garch_recursion",
    "garch_sim",
    "get_distribution",
    "gjr_recursion",
    "gjr_sim",
    "initial_variance",
    "loggarch_recursion",
    "loggarch_sim",
    "megarch_recursion",
    "megarch_sim",
    "mloggarch_recursion",
    "mloggarch_sim",
    "quasi_max_likelihood",
    "tgarch_recursion",
    "tgarch_sim",
    "theta_coefficients",
    "type1_news_impact",
]
