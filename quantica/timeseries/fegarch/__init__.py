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

**Phase 3 — the fractional-differencing engine** (complete):

* **(1-L)^d operator** (:mod:`~quantica.timeseries.fegarch.fracdiff`) — the binomial
  :math:`(1-L)^d = \sum_i (-1)^i \binom{d}{i} L^i` filter (Hosking 1981; Bollerslev-Mikkelsen 1996),
  :func:`~quantica.timeseries.fegarch.fracdiff_coeffs` +
  :func:`~quantica.timeseries.fegarch.fracdiff`
  (direct and FFT paths, App. C.3 truncation ``L = n-1`` / pre-sample 0). An internal filter, not a
  fitted model — validated analytically + cross-method, the load-bearing infrastructure for Phase 4.

**Phase 4 — long-memory (fractionally-integrated) models** (Type-I family complete):

* **FIEGARCH / FIMEGARCH / FIMLog-GARCH** (:mod:`~quantica.timeseries.fegarch.fiegarch`) — the
  fractionally-integrated Type-I EGF models, a truncated-MA(∞) log-variance recursion
  :math:`\ln\sigma_t^2 = \omega_\sigma + \sum_i \theta_i\, g(\eta_{t-1-i})` whose coefficients
  :math:`\theta(B) = \phi^{-1}(B)(1-B)^{-d}\psi(B)` splice the Phase-3 fractional operator into the
  Phase-2 Type-I news impact :math:`g`. They are built by composition — the same
  :func:`~quantica.timeseries.fegarch.theta_coefficients` machinery, differing only in the
  constant-set (EGARCH / MEGARCH / MLog-GARCH) — via
  :func:`~quantica.timeseries.fegarch.fit_fiegarch` /
  :func:`~quantica.timeseries.fegarch.fit_fimegarch` /
  :func:`~quantica.timeseries.fegarch.fit_fimloggarch` and their simulators. The fractional order
  :math:`d \in (0, 1)` is estimated (not clamped at ``0.5``).
* **FIGARCH(1,d,1)** (:mod:`~quantica.timeseries.fegarch.figarch`) — the first
  **variance-recursion** long-memory model (a different seam from the EGF family): the operator
  enters the conditional-variance polynomial,
  :math:`\sigma_t^2 = \omega + \sum_i \theta_i \varepsilon_{t-i}^2`
  with :math:`\theta(B) = 1-(1-\phi_1 B)(1-B)^d/(1-\beta_1 B)` (Baillie-Bollerslev-Mikkelsen 1996;
  WP175). The intercept :math:`\omega` is used directly, and — with no :math:`\eta`/:math:`\sigma^2`
  feedback — :math:`\sigma_t^2` is a pure linear filter of observed :math:`\varepsilon^2`. Via
  :func:`~quantica.timeseries.fegarch.fit_figarch` and
  :func:`~quantica.timeseries.fegarch.figarch_sim`; the shared primitives
  :func:`~quantica.timeseries.fegarch.figarch_coefficients` and
  :func:`~quantica.timeseries.fegarch.figarch_variance_filter` (with the resolved 50-term
  ``Var(ddof=1)`` pre-sample) are inherited by the FIAPARCH / FITGARCH / FIGJR group.
* **FIAPARCH(1,d,1)** (:mod:`~quantica.timeseries.fegarch.fiaparch`) — FIGARCH's variance-recursion
  seam with the APARCH power-asymmetry news,
  :math:`\sigma_t^\delta = \omega + \sum_i \theta_i (|\varepsilon|-\gamma\varepsilon)^\delta_{t-i}`,
  using the *same* :math:`\theta(B)` as FIGARCH (Tse 1998; WP175 Eq. 2.10). Reuses
  :func:`~quantica.timeseries.fegarch.figarch_coefficients` and
  :func:`~quantica.timeseries.fegarch.figarch_variance_filter` with the news swapped, via
  :func:`~quantica.timeseries.fegarch.fit_fiaparch` /
  :func:`~quantica.timeseries.fegarch.fiaparch_sim`. The δ-power seam is machine-exact, but the
  presample seed is a **documented bounded limit** (the 2nd irreducible-from-output value after the
  APARCH :math:`\sigma_0` fork) — seeded at ``mean(news)`` with the residual documented.
* **FITGARCH / FIGJR(1,d,1)** (:mod:`~quantica.timeseries.fegarch.fiaparch`) — FIAPARCH with the
  power :math:`\delta` **fixed**: FITGARCH at :math:`\delta=1` (Zakoian 1994) and FIGJR at
  :math:`\delta=2` (Glosten et al. 1993). Thin wrappers over the FIAPARCH machinery (bit-for-bit
  ``fiaparch_recursion`` at fixed δ), fitting ``{mu, omega, phi1, beta1, gamma, d}`` via
  :func:`~quantica.timeseries.fegarch.fit_fitgarch` / :func:`~quantica.timeseries.fegarch.fit_figjr`
  and their simulators. The reconstruction gate confirmed the **FIGJR kernel is
  :math:`(|\varepsilon|-\gamma\varepsilon)^2`, not the Glosten indicator** (the Phase-1 GJR finding
  in FI form). This completes the variance-recursion FI family.
* **FILog-GARCH(1,d,1)** (:mod:`~quantica.timeseries.fegarch.filoggarch`) — the **Type-II**
  fractional model (fractionally-integrated Log-GARCH, the counterpart of FIEGARCH). Its MA(∞) loads
  the log-square news :math:`\xi = \ln\eta^2 - \operatorname{E}[\ln\eta^2]` through
  :math:`\gamma(B) = (1-\phi_1 B)^{-1}(1-B)^{-d}(1+\psi_1 B) - 1` (WP 2026-04 Eqs. 10-11), reusing
  FIEGARCH's :func:`~quantica.timeseries.fegarch.theta_coefficients` + the :math:`(1+\psi_1 B)` MA
  factor and the Log-GARCH ``mean_log_sq`` moment, via
  :func:`~quantica.timeseries.fegarch.fit_filoggarch` /
  :func:`~quantica.timeseries.fegarch.filoggarch_sim`. The seam is machine-exact; the fractional
  :math:`d` relaxes short-memory Log-GARCH's near-common-root ridge. **This is Phase 4's clearest
  effective-challenge result**: fEGarch's committed fixture fit on the synthetic series is
  non-converged, and an independent clean-room fit reaches a ~+120-higher log-likelihood.

**Phase 4 is complete** — all eight fractionally-integrated models. The dual mean and the
forecasting/risk tie-back arrive in later phases — see ``docs/fegarch-port-roadmap.md``.
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
from quantica.timeseries.fegarch.fiaparch import (
    fiaparch_news,
    fiaparch_recursion,
    fiaparch_sim,
    figjr_recursion,
    figjr_sim,
    fit_fiaparch,
    fit_figjr,
    fit_fitgarch,
    fitgarch_recursion,
    fitgarch_sim,
)
from quantica.timeseries.fegarch.fiegarch import (
    fiegarch_recursion,
    fiegarch_sim,
    fimegarch_recursion,
    fimegarch_sim,
    fimloggarch_recursion,
    fimloggarch_sim,
    fit_fiegarch,
    fit_fimegarch,
    fit_fimloggarch,
    theta_coefficients,
)
from quantica.timeseries.fegarch.figarch import (
    FIGARCH_PRESAMPLE,
    figarch_coefficients,
    figarch_recursion,
    figarch_sim,
    figarch_variance_filter,
    fit_figarch,
)
from quantica.timeseries.fegarch.filoggarch import (
    filoggarch_gamma_coefficients,
    filoggarch_recursion,
    filoggarch_sim,
    fit_filoggarch,
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
    "FIGARCH_PRESAMPLE",
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
    "fiaparch_news",
    "fiaparch_recursion",
    "fiaparch_sim",
    "fiegarch_recursion",
    "fiegarch_sim",
    "figarch_coefficients",
    "figarch_recursion",
    "figarch_sim",
    "figarch_variance_filter",
    "figjr_recursion",
    "figjr_sim",
    "filoggarch_gamma_coefficients",
    "filoggarch_recursion",
    "filoggarch_sim",
    "fimegarch_recursion",
    "fimegarch_sim",
    "fimloggarch_recursion",
    "fimloggarch_sim",
    "fit_aparch",
    "fit_egarch",
    "fit_fiaparch",
    "fit_fiegarch",
    "fit_figarch",
    "fit_figjr",
    "fit_filoggarch",
    "fit_fimegarch",
    "fit_fimloggarch",
    "fit_fitgarch",
    "fit_garch",
    "fit_gjr",
    "fit_loggarch",
    "fit_megarch",
    "fit_mloggarch",
    "fit_tgarch",
    "fitgarch_recursion",
    "fitgarch_sim",
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
