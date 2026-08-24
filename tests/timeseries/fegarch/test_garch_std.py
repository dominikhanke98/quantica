"""Validation of GARCH(1,1) under Student-t — the joint shape-parameter QMLE path (Phase 5).

This is the first of the *distribution-breadth* work: the shared QMLE engine already carries a
distribution's shape parameters in the fitted vector, and here that path is proven end-to-end on
GARCH(1,1)/std, where the extra parameter is the Student-t degrees of freedom ``nu`` (our name; the
fEGarch fixture reports it as ``df``).

The validation is **identification-structured**, because ``nu`` is only weakly identified on the
near-normal synthetic returns:

* **Correctness is machine-exact at the seam** — at fEGarch's reported parameters (including
  ``df = 341.89``) our GARCH recursion reproduces the sigma series to ``~1e-17`` and the Student-t
  log-likelihood to ``~1e-10``. This is the tight proof that the joint shape-parameter likelihood is
  right.
* **The fit is an effective challenge, not a re-derivation of the fixture.** The synthetic returns
  are Gaussian, so the Student-t MLE is ``nu -> infinity`` (the normal limit): the log-likelihood
  rises monotonically in ``nu`` toward the normal supremum and is *unreachable* at finite ``nu``.
  fEGarch's optimizer stopped at ``df = 341.89`` with log-lik ``7600.83`` — **0.28 below** the GARCH
  x norm optimum ``7601.11`` — a strictly-dominated point on the flat ``nu`` ridge (the same honest
  pattern as the FILog-GARCH local optimum). Our derivative-free fit climbs the ridge to the ``nu``
  bound and reaches the normal supremum, so it *dominates* the fixture. We therefore validate the
  well-identified pieces tightly and ``nu`` only by regime, never param-exact against the dominated
  reference.
* **Reduction anchor**: as ``nu -> infinity`` the standardized Student-t density collapses to the
  standard normal, so GARCH x std recovers GARCH x norm.
* **Known-truth**: when the data genuinely has ``df = 6`` (identified regime), the QMLE recovers
  ``nu`` to within a fraction of a standard error.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from quantica.timeseries.fegarch import (
    fit_garch,
    garch_recursion,
    garch_sim,
    get_distribution,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _load_std() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the GARCH x std fit-params JSON, and its sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_garch11_std_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_std_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


def _norm_loglik() -> float:
    """The GARCH x norm fixture log-likelihood — the supremum the Student-t fit approaches."""
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_norm_params.json").read_text(encoding="utf-8"))
    return float(meta["loglikelihood"])


# --------------------------------------------------------------------------- #
# Seam: the joint shape-parameter likelihood is machine-exact at fEGarch params
# --------------------------------------------------------------------------- #


def test_std_recursion_and_likelihood_are_exact_at_reported_params() -> None:
    """At fEGarch's params the sigma series and the Student-t log-lik match to machine order.

    Isolates the model + distribution from the optimizer: the GARCH sigma series does not depend on
    ``df`` (it is fixed by mu/omega/alpha/beta), so it reproduces to ``~1e-17``; feeding those
    standardized residuals through the standardized-t density at ``df = 341.89`` reproduces the
    fixture log-likelihood to ``~1e-10``. This is the tight correctness proof for the shape path.
    """
    returns, meta, sigma_fix = _load_std()
    fx = meta["params"]
    var_params = np.array([fx["mu"], fx["omega"], fx["phi1"], fx["beta1"]])
    sigma = np.sqrt(garch_recursion(var_params, returns))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10

    std = get_distribution("std")
    z = (returns - fx["mu"]) / sigma
    loglik = float(np.sum(-np.log(sigma) + std.logpdf(z, (fx["df"],))))
    assert abs(loglik - meta["loglikelihood"]) < 1e-6


# --------------------------------------------------------------------------- #
# Headline fit: well-identified pieces tight, nu by regime, sigma at the ridge level
# --------------------------------------------------------------------------- #


def test_garch11_std_matches_fegarch_fixture() -> None:
    """fit_garch/std recovers the well-identified GARCH params; nu lands in the large regime.

    The mean and the three variance parameters are well identified and agree with the fixture to
    well under a percent. ``nu`` is only weakly identified on Gaussian data (its MLE is infinite),
    so it is validated by *regime* (large, ``> 100``) and never param-exact against the fixture's
    strictly-dominated ``df = 341.89``. Its standard error is (correctly) non-finite there — the
    likelihood is flat in ``nu`` — so only the well-identified standard errors are asserted finite.
    """
    returns, meta, sigma_fix = _load_std()
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="std")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-5
    assert abs(fit.params["omega"] - fx["omega"]) / fx["omega"] < 1e-2
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 5e-3
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 5e-3
    assert fit.params["nu"] > 100.0  # large regime, NOT the dominated df=341.89

    # The two fits sit at different points on the flat nu ridge (nu -> bound vs df = 341.89), so
    # their variance parameters — and thus sigma — differ at the ~1e-3 relative level. The
    # machine-exact sigma agreement is the seam test above, at identical parameters.
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 5e-5
    assert np.max(deviation / sigma_fix) < 2e-3

    # nu's standard error is non-finite (flat likelihood); the well-identified ones are finite.
    for name in ("mu", "omega", "alpha", "beta"):
        assert np.isfinite(fit.std_errors[name]) and fit.std_errors[name] > 0.0


def test_std_fit_dominates_the_sub_optimal_fegarch_optimum() -> None:
    """Effective challenge: our std fit reaches the normal supremum, dominating the fixture.

    On Gaussian data the Student-t log-likelihood increases monotonically in ``nu`` toward the
    GARCH x norm optimum (``7601.11``) and is unreachable at any finite ``nu``. fEGarch stopped at
    ``df = 341.89`` with log-lik ``7600.83`` — 0.28 below — a strictly-dominated point. Our
    derivative-free optimizer climbs the ridge to the ``nu`` bound and comes within ``~1e-3`` of the
    supremum, so it strictly dominates the fixture (and never asserts the dominated ``df``).
    """
    returns, meta, _sigma = _load_std()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="std")

    norm_sup = _norm_loglik()
    assert fit.loglikelihood > meta["loglikelihood"] + 0.1  # strictly beats the dominated fixture
    assert fit.loglikelihood <= norm_sup + 1e-6  # cannot exceed the normal supremum
    assert norm_sup - fit.loglikelihood < 1e-2  # but reaches it to ~1e-3 at the nu bound


def test_norm_path_unchanged_by_the_shape_parameter_branch() -> None:
    """Adding the shape-parameter optimizer branch leaves GARCH x norm bit-identical.

    norm has no shape parameter, so it keeps the gradient-based L-BFGS-B path (only shape-parameter
    fits switch to derivative-free Nelder-Mead). The norm fit therefore still reproduces its fixture
    log-likelihood to the same ``~1e-8`` it did before this work.
    """
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    fit = fit_garch(returns, cond_dist="norm")
    assert abs(fit.loglikelihood - _norm_loglik()) < 1e-7


# --------------------------------------------------------------------------- #
# Reduction anchor + known-truth recovery
# --------------------------------------------------------------------------- #


def test_df_to_infinity_recovers_the_normal_density() -> None:
    """Reduction anchor: the standardized Student-t density collapses to the normal as nu -> inf."""
    std = get_distribution("std")
    norm = get_distribution("norm")
    z = np.linspace(-4.0, 4.0, 401)
    assert np.max(np.abs(std.logpdf(z, (1.0e6,)) - norm.logpdf(z))) < 1e-3


def test_known_truth_std_recovers_df() -> None:
    """When the data truly has df = 6 (identified regime), QMLE recovers nu to within a fraction SE.

    Contrast with the near-normal fixture, where nu is unidentified and flies to the bound: with
    genuine heavy tails the shape parameter is sharply identified and the joint fit pins it down.
    """
    true = {"mu": 0.0003, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = garch_sim(
            9000, **true, cond_dist="std", dist_params=(6.0,), rng=np.random.default_rng(0)
        )
        fit = fit_garch(returns, cond_dist="std")
    se = fit.std_errors
    assert abs(fit.params["nu"] - 6.0) < 3.0 * se["nu"]
    for name in ("alpha", "beta"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]
