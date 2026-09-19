"""ARMA-in-mean x GARCH(1,1) dual fits -- the first non-constant mean (Phase 5).

The joint mean+variance QMLE: an ARMA(P,Q) conditional mean estimated together with the GARCH(1,1)
variance in one likelihood. Validation follows the FIAPARCH §19 bounded-limit-seed pattern -- the
**seam** (recursion + coupling) is asserted machine-exact separately from the fitted match, because
the dual-case ``sigma_0^2`` seed is an internal fEGarch preliminary-residual quantity not derivable
from published math (§12); we use the reduction-consistent ``Var(r, ddof=1)`` analog and document
the resulting decaying seed transient as a bound.

* **Seam machine-exact**: at the fixture's own params + its implied ``sigma_0^2`` the recursion +
  coupling reproduce ``sigma`` to ``~1e-15`` (6.9e-18) -- proof the mean recursion, the coupling,
  and the variance recursion are exactly fEGarch's.
* **Bounded seed**: with the ``Var(r, ddof=1)`` seed the reconstruction leaves only a decaying
  transient (< 1e-5 at t=0, machine-zero in the tail).
* **Fitted match**: ARMA(1,0)/(0,1) are identified and match tightly; ARMA(1,1) is near-common-root
  (AR/MA roots nearly cancel on mean-less data), so our optimizer **dominates** fEGarch's fixture at
  a different ``(ar1, ma1)`` -- validated by loglik + the variance block, not a mean-param match.
* **Reduction**: ARMA(0,0) reproduces ``fit_garch`` exactly (the seed collapses to ``Var(y)``).
* **Known-truth**: a simulated ARMA(1,1)-GARCH with identified ``ar1=0.5, ma1=0.3`` recovers them --
  the positive control that mean estimation works when the data has genuine mean dynamics.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import fit_arma_garch, fit_garch
from quantica.timeseries.fegarch.garch import _garch_variance
from quantica.timeseries.fegarch.mean import arma_mean_residuals

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# tag -> (P, Q); the committed dual fixtures are norm-only.
_ORDERS = {"arma10": (1, 0), "arma01": (0, 1), "arma11": (1, 1)}
_IDENTIFIED = [
    "arma10",
    "arma01",
]  # single-term means are identified (ARMA(1,1) is near-common-root)


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _load(tag: str) -> tuple[dict, np.ndarray]:  # type: ignore[type-arg]
    """Load a dual fixture's params dict and sigma series."""
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_{tag}_garch11_norm_params.json").read_text(encoding="utf-8")
    )
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_{tag}_garch11_norm_sigma.csv", skiprows=1)
    return meta, sigma


# --------------------------------------------------------------------------- #
# The mean recursion (form + reduction)
# --------------------------------------------------------------------------- #


def test_arma_mean_residuals_form_and_reduction() -> None:
    """ARMA(0,0) gives exactly y-mu; the ARMA(1,1) recursion matches hand-computed first r_t."""
    y = _returns()
    mu = 0.0005
    # ARMA(0,0): r_t = y_t - mu, exactly.
    assert np.max(np.abs(arma_mean_residuals(y, mu, 0.0, 0.0) - (y - mu))) == 0.0
    # ARMA(1,1) explicit recursion (0 pre-sample): r_0 = y_0-mu; r_1 = y_1-mu-ar1(y_0-mu)-ma1 r_0.
    ar1, ma1 = 0.3, -0.2
    r = arma_mean_residuals(y, mu, ar1, ma1)
    assert r[0] == pytest.approx(y[0] - mu, abs=1e-18)
    r1_hand = y[1] - (mu + ar1 * (y[0] - mu) + ma1 * (y[0] - mu))  # r_0 = y_0-mu
    assert r[1] == pytest.approx(r1_hand, abs=1e-18)


# --------------------------------------------------------------------------- #
# Seam machine-exact (fixture params + implied seed) — the correctness proof
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tag", ["arma10", "arma01", "arma11"])
def test_seam_machine_exact(tag: str) -> None:
    """At the fixture params + its implied sigma_0^2, recursion+coupling rebuild sigma to ~1e-15.

    This isolates the recursion and the mean-variance coupling from the seed: the ARMA residuals
    drive the existing GARCH variance recursion, and seeded from the fixture's own sigma_0^2 the
    whole series pins to machine order -- the mean recursion and coupling are exactly fEGarch's.
    """
    meta, sigma_fix = _load(tag)
    p = meta["params"]
    r = arma_mean_residuals(_returns(), p["mu"], p.get("ar1", 0.0), p.get("ma1", 0.0))
    implied_seed = (sigma_fix[0] ** 2 - p["omega"]) / (p["phi1"] + p["beta1"])
    sigma = np.sqrt(_garch_variance(r, p["omega"], p["phi1"], p["beta1"], implied_seed))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-14


@pytest.mark.parametrize("tag", ["arma10", "arma01", "arma11"])
def test_bounded_seed_transient(tag: str) -> None:
    """The Var(r,ddof=1) bounded seed leaves only a decaying transient (<1e-5, machine-0 tail).

    fEGarch's exact dual seed is an internal preliminary-residual quantity (non-standard denominator
    ~ n-1.4) that no published-math form reproduces (§12); the Var(r) analog reduces to the
    pure-GARCH Var(y) seed at ARMA(0,0) and its seed error is a bounded, decaying transient.
    """
    meta, sigma_fix = _load(tag)
    p = meta["params"]
    r = arma_mean_residuals(_returns(), p["mu"], p.get("ar1", 0.0), p.get("ma1", 0.0))
    seed = p["omega"] + (p["phi1"] + p["beta1"]) * float(np.var(r, ddof=1))
    dev = np.abs(np.sqrt(_garch_variance(r, p["omega"], p["phi1"], p["beta1"], seed)) - sigma_fix)
    assert dev.max() < 1e-5  # the documented seed-transient bound
    assert dev[500:].max() < 1e-12  # decays to machine-zero away from the seed


# --------------------------------------------------------------------------- #
# Fitted match: ARMA(1,0)/(0,1) tight; ARMA(1,1) dominates (near-common-root)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tag", _IDENTIFIED)
def test_identified_mean_fitted_match(tag: str) -> None:
    """ARMA(1,0)/(0,1): the single-term mean is identified -- loglik, variance, ar1/ma1 all tight.

    The bounded Var(r) seed leaves a negligible loglik effect (< 1e-3) and a sigma deviation < 1e-5;
    the variance block matches, and the lone mean coefficient recovers fEGarch's value.
    """
    p_order, q_order = _ORDERS[tag]
    meta, sigma_fix = _load(tag)
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_arma_garch(_returns(), mean_orders=(p_order, q_order), cond_dist="norm")
    assert fit.converged
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-3  # bounded-seed loglik effect
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 5e-3  # variance block tight
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 5e-3
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5  # bounded seed
    if p_order:
        assert abs(fit.params["ar1"] - fx["ar1"]) < 5e-3  # identified -> recovers
    if q_order:
        assert abs(fit.params["ma1"] - fx["ma1"]) < 5e-3


def test_arma11_dominates_the_near_common_root_fixture() -> None:
    """ARMA(1,1): AR/MA roots nearly cancel on mean-less data -- our fit dominates the fixture.

    The variance block still matches tightly, but the mean block sits on a flat near-common-root
    ridge where fEGarch's Hessian fails; our optimizer lands a comparable-or-higher likelihood
    (~ +0.06) at a different (ar1, ma1). Validated by loglik-dominates + the variance block, not a
    mean-param match -- the §14/§21 weak-identification pattern on the ARMA mean.
    """
    meta, _sigma = _load("arma11")
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_arma_garch(_returns(), mean_orders=(1, 1), cond_dist="norm")
    assert fit.converged
    assert fit.loglikelihood >= meta["loglikelihood"] - 1e-4  # reaches or dominates the ridge
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 5e-3  # variance block still tight
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 5e-3


# --------------------------------------------------------------------------- #
# Reduction anchor + known-truth
# --------------------------------------------------------------------------- #


def test_reduction_to_constant_mean_garch() -> None:
    """ARMA(0,0) reproduces fit_garch exactly: the mean recursion generalizes the constant mean.

    With no ARMA terms mu_t = mu (constant), the mean-residuals are exactly y-mu and the Var(r) seed
    collapses to Var(y) -- so the dual fit and fit_garch coincide to machine order.
    """
    returns = _returns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dual = fit_arma_garch(returns, mean_orders=(0, 0), cond_dist="norm")
        base = fit_garch(returns, cond_dist="norm")
    assert abs(dual.loglikelihood - base.loglikelihood) < 1e-10
    assert np.max(np.abs(dual.conditional_volatility - base.conditional_volatility)) < 1e-12


def _sim_arma_garch(  # type: ignore[no-untyped-def]
    n, *, mu, ar1, ma1, omega, alpha, beta, rng, n_burn=1000
):
    """Simulate an ARMA(1,1)-GARCH(1,1) path (norm innovations) for the known-truth control."""
    total = n + n_burn
    z = rng.standard_normal(total)
    r = np.empty(total)
    y = np.empty(total)
    s2 = np.empty(total)
    uncond = omega / (1.0 - alpha - beta)
    for t in range(total):
        prev_r2 = r[t - 1] ** 2 if t else uncond
        prev_s2 = s2[t - 1] if t else uncond
        s2[t] = omega + alpha * prev_r2 + beta * prev_s2
        r[t] = np.sqrt(s2[t]) * z[t]
        ydev_prev = (y[t - 1] - mu) if t else 0.0
        r_prev = r[t - 1] if t else 0.0
        y[t] = mu + ar1 * ydev_prev + ma1 * r_prev + r[t]
    return np.asarray(y[n_burn:], dtype=np.float64)


def test_known_truth_recovers_identified_mean() -> None:
    """A simulated ARMA(1,1)-GARCH with identified ar1=0.5, ma1=0.3 recovers all four in a few SE.

    The positive control (separate from the weakly-identified synthetic fixture): when the data has
    genuine mean dynamics, the joint QMLE recovers the ARMA mean coefficients and the variance.
    """
    true = {"mu": 0.0003, "ar1": 0.5, "ma1": 0.3, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    y = _sim_arma_garch(8000, **true, rng=np.random.default_rng(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_arma_garch(y, mean_orders=(1, 1), cond_dist="norm")
    assert fit.converged
    se = fit.std_errors
    for name in ("ar1", "ma1", "alpha", "beta"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


# --------------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------------- #


def test_invalid_mean_orders_and_unsupported_dist() -> None:
    """mean_orders outside {0,1} raise; ald/sald (discrete-P profiling) are not yet supported."""
    y = _returns()
    with pytest.raises(ValueError, match="must each be 0 or 1"):
        fit_arma_garch(y, mean_orders=(2, 0))
    with pytest.raises(NotImplementedError, match="discrete-P profiling"):
        fit_arma_garch(y, mean_orders=(1, 1), cond_dist="ald")
