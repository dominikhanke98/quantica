"""Validation of the fEGarch asymmetric short-memory models (numerical-validation skill, Phase 1).

GJR-GARCH, TGARCH and APARCH are reconciled against the committed fEGarch fit fixtures. The headline
per model is the **fEGarch-fit match**: fitting the model to the committed synthetic returns
reproduces fEGarch's parameters, log-likelihood, information criteria and conditional-SD series to
tolerance. The reconciliation showed all three are the single APARCH power recursion at
``delta in {2, 1, free}`` (GJR/TGARCH/APARCH); the recursion form is machine-exact and the residual
is a pre-sample effect (largest for the free-delta APARCH, whose realized deviations are documented
here honestly). Reduction anchors confirm the model nesting, and known-truth simulations confirm
QMLE recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
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

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _load(name: str) -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_{name}_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / f"fit_{name}_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


# --------------------------------------------------------------------------- #
# Headline fixture matches (fEGarch names: mu, omega, phi1, beta1, gamma1[, delta])
# --------------------------------------------------------------------------- #


def test_gjr_matches_fegarch_fixture() -> None:
    """fit_gjr reproduces fEGarch's GJR-GARCH(1,1)/norm fit to GARCH-level tolerance."""
    returns, meta, sigma_fix = _load("gjrgarch11_norm")
    fx = meta["params"]
    fit = fit_gjr(returns, cond_dist="norm")
    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega", "phi1", "beta1", "gamma1"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_tgarch_matches_fegarch_fixture() -> None:
    """fit_tgarch reproduces fEGarch's TGARCH(1,1)/norm fit to GARCH-level tolerance."""
    returns, meta, sigma_fix = _load("tgarch11_norm")
    fx = meta["params"]
    fit = fit_tgarch(returns, cond_dist="norm")
    # omega ~ 2.3e-4 (sigma-units), an order of magnitude above GJR's ~3e-6 (sigma^2-units).
    assert 1e-4 < fit.params["omega"] < 1e-3
    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega", "phi1", "beta1", "gamma1"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_aparch_matches_fegarch_fixture() -> None:
    """fit_aparch reproduces fEGarch's APARCH(1,1)/norm fit, incl. the free power ``delta``.

    APARCH carries a documented pre-sample residual (the delta-th absolute moment does not exactly
    reproduce fEGarch's unpublished pre-sample state at the fitted ``delta ~ 2.41``); the tolerances
    below are the realized deviations. The recursion form itself is machine-exact (see
    ``test_recursion_form_is_exact_seeded_from_fixture``), and ``delta`` and the well-identified
    parameters still match tightly.
    """
    returns, meta, sigma_fix = _load("aparch11_norm")
    fx = meta["params"]
    fit = fit_aparch(returns, cond_dist="norm")
    assert abs(fit.params["mu"] - fx["mu"]) < 1e-5
    assert (
        abs(fit.params["delta"] - fx["delta"]) / fx["delta"] < 3e-3
    )  # free power, well identified
    assert abs(fit.params["beta1"] - fx["beta1"]) / fx["beta1"] < 1e-3
    assert abs(fit.params["gamma1"] - fx["gamma1"]) / abs(fx["gamma1"]) < 1e-3
    assert abs(fit.params["phi1"] - fx["phi1"]) / fx["phi1"] < 3e-3
    assert abs(fit.params["omega"] - fx["omega"]) / fx["omega"] < 3e-2  # weakly identified
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 2e-2
    assert abs(fit.aic - meta["aic"]) < 1e-4
    assert abs(fit.bic - meta["bic"]) < 1e-4
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 2e-4
    assert np.max(deviation / sigma_fix) < 1e-2
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


@pytest.mark.parametrize(
    ("name", "recursion", "delta"),
    [
        ("gjrgarch11_norm", gjr_recursion, 2.0),
        ("tgarch11_norm", tgarch_recursion, 1.0),
        ("aparch11_norm", aparch_recursion, None),
    ],
)
def test_recursion_form_is_exact_seeded_from_fixture(name, recursion, delta) -> None:  # type: ignore[no-untyped-def]
    """At fEGarch's own params, the recursion reproduces sigma[1:] to machine precision.

    This isolates the recursion *form* from the pre-sample: seeding sigma[0] from the fixture and
    stepping forward with the reported parameters matches the whole tail to ~1e-15, for all three
    models — confirming the shared ``(|eps| - gamma*eps)^delta`` kernel is exactly fEGarch's.
    """
    returns, meta, sigma_fix = _load(name)
    p = meta["params"]
    d = p["delta"] if delta is None else delta
    if delta is None:
        params = np.array([p["mu"], p["omega"], p["phi1"], p["beta1"], p["gamma1"], p["delta"]])
    else:
        params = np.array([p["mu"], p["omega"], p["phi1"], p["beta1"], p["gamma1"]])
    # Rebuild sigma^2 with the recursion but overwrite sigma[0] from the fixture, then step forward.
    resid = returns - p["mu"]
    sig_delta = np.empty(returns.size)
    sig_delta[0] = sigma_fix[0] ** d
    for t in range(1, returns.size):
        kernel = (abs(resid[t - 1]) - p["gamma1"] * resid[t - 1]) ** d
        sig_delta[t] = p["omega"] + p["phi1"] * kernel + p["beta1"] * sig_delta[t - 1]
    sigma = sig_delta ** (1.0 / d)
    assert np.max(np.abs(sigma[1:] - sigma_fix[1:])) < 1e-12
    # sanity: the public recursion returns a positive variance path of the right shape
    assert recursion(params, returns).shape == returns.shape


# --------------------------------------------------------------------------- #
# Reduction anchors: the APARCH family nests GJR and GARCH
# --------------------------------------------------------------------------- #


def test_reduction_anchors() -> None:
    """gamma1=0 collapses GJR to GARCH; delta=2 collapses APARCH to GJR (structurally exact)."""
    rng = np.random.default_rng(7)
    returns, _sig = gjr_sim(2000, omega=3e-6, phi1=0.08, beta1=0.9, gamma1=0.05, rng=rng)

    # gamma1 = 0: GJR kernel (|eps|)^2 == eps^2, the plain GARCH news impact.
    from quantica.timeseries.fegarch import garch_recursion

    gjr_sym = gjr_recursion(np.array([0.0, 3e-6, 0.08, 0.9, 0.0]), returns)
    garch = garch_recursion(np.array([0.0, 3e-6, 0.08, 0.9]), returns)
    # Same recursion at gamma=0; pre-sample kernels differ only by ddof (mean|eps|^2 vs Var), tiny.
    assert np.max(np.abs(gjr_sym - garch)) < 1e-5

    # delta = 2: APARCH == GJR for identical (mu, omega, phi1, beta1, gamma1).
    gjr = gjr_recursion(np.array([0.0, 3e-6, 0.08, 0.9, 0.05]), returns)
    aparch_d2 = aparch_recursion(np.array([0.0, 3e-6, 0.08, 0.9, 0.05, 2.0]), returns)
    assert np.max(np.abs(aparch_d2 - gjr)) == 0.0


# --------------------------------------------------------------------------- #
# Known-truth recovery + simulation invariants
# --------------------------------------------------------------------------- #


def test_gjr_known_truth_recovery() -> None:
    """QMLE recovers planted GJR (phi1, beta1, gamma1) from a simulated series within a few SE."""
    true = {"mu": 0.0002, "omega": 3e-6, "phi1": 0.04, "beta1": 0.90, "gamma1": 0.08}
    returns, _sigma = gjr_sim(9000, **true, rng=np.random.default_rng(0))
    fit = fit_gjr(returns, cond_dist="norm")
    se = fit.std_errors
    assert abs(fit.params["phi1"] - true["phi1"]) < 4.0 * se["phi1"]
    assert abs(fit.params["beta1"] - true["beta1"]) < 4.0 * se["beta1"]
    assert abs(fit.params["gamma1"] - true["gamma1"]) < 4.0 * se["gamma1"]


def test_tgarch_known_truth_recovery() -> None:
    """QMLE recovers planted TGARCH (phi1, beta1, gamma1) within a few SE."""
    true = {"mu": 0.0, "omega": 5e-4, "phi1": 0.08, "beta1": 0.90, "gamma1": 0.10}
    returns, _sigma = tgarch_sim(9000, **true, rng=np.random.default_rng(1))
    fit = fit_tgarch(returns, cond_dist="norm")
    se = fit.std_errors
    assert abs(fit.params["phi1"] - true["phi1"]) < 4.0 * se["phi1"]
    assert abs(fit.params["beta1"] - true["beta1"]) < 4.0 * se["beta1"]
    assert abs(fit.params["gamma1"] - true["gamma1"]) < 4.0 * se["gamma1"]


def test_aparch_known_truth_recovery() -> None:
    """QMLE recovers planted APARCH (beta1, gamma1, delta) within a few SE; power is identified."""
    true = {"mu": 0.0, "omega": 5e-6, "phi1": 0.06, "beta1": 0.90, "gamma1": 0.10, "delta": 1.6}
    returns, _sigma = aparch_sim(12000, **true, rng=np.random.default_rng(2))
    fit = fit_aparch(returns, cond_dist="norm")
    se = fit.std_errors
    assert abs(fit.params["beta1"] - true["beta1"]) < 4.0 * se["beta1"]
    assert abs(fit.params["gamma1"] - true["gamma1"]) < 4.0 * se["gamma1"]
    assert abs(fit.params["delta"] - true["delta"]) < 4.0 * se["delta"]


def test_sims_are_positive_and_shaped() -> None:
    """All three simulators return positive sigma paths of the requested length."""
    rng = np.random.default_rng(3)
    for returns, sigma in (
        gjr_sim(3000, omega=3e-6, phi1=0.05, beta1=0.92, gamma1=0.04, rng=rng),
        tgarch_sim(3000, omega=5e-4, phi1=0.07, beta1=0.90, gamma1=0.10, rng=rng),
        aparch_sim(3000, omega=5e-6, phi1=0.05, beta1=0.90, gamma1=0.08, delta=1.4, rng=rng),
    ):
        assert returns.shape == sigma.shape == (3000,)
        assert np.all(sigma > 0.0)


def test_aparch_sim_supports_heavy_tails() -> None:
    """Simulation routes through the Phase-0 distribution layer (a heavy-tailed Student-t)."""
    returns, sigma = aparch_sim(
        2500,
        omega=5e-6,
        phi1=0.06,
        beta1=0.88,
        gamma1=0.10,
        delta=1.5,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(4),
    )
    assert returns.shape == sigma.shape == (2500,)
    assert np.all(sigma > 0.0)


def test_sims_reject_bad_inputs() -> None:
    """Simulators reject non-positive size, |gamma1| >= 1, delta <= 0 and non-stationary specs."""
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="n must be positive"):
        gjr_sim(0, omega=3e-6, phi1=0.05, beta1=0.9, gamma1=0.04, rng=rng)
    with pytest.raises(ValueError, match=r"\|gamma1\| < 1"):
        tgarch_sim(100, omega=5e-4, phi1=0.05, beta1=0.9, gamma1=1.0, rng=rng)
    with pytest.raises(ValueError, match="delta > 0"):
        aparch_sim(100, omega=5e-6, phi1=0.05, beta1=0.9, gamma1=0.1, delta=0.0, rng=rng)
    with pytest.raises(ValueError, match="stationarity"):
        gjr_sim(100, omega=3e-6, phi1=0.6, beta1=0.9, gamma1=0.5, rng=rng)


# --------------------------------------------------------------------------- #
# Cross-check vs the `arch` package (skip-safe; CI validates it)
# --------------------------------------------------------------------------- #


def test_gjr_cross_check_against_arch() -> None:
    """Our GJR-GARCH agrees with ``arch``'s GJR (``o=1``) on the conditional-volatility path.

    The coefficients are not directly comparable — ``arch`` uses the Glosten indicator
    parameterization while fEGarch (and we) use the ``(|eps| - gamma*eps)^2`` kernel — but both fit
    a GJR-type variance to the same data, so their conditional-volatility paths are near-identical
    (correlation > 0.999) and both detect a positive leverage effect. Skipped where ``arch`` (or its
    ``numba``/``llvmlite`` backend) cannot load; CI validates it.
    """
    try:
        from arch import arch_model
    except Exception as exc:  # numba/llvmlite can fail to load outside CI
        pytest.skip(f"arch unavailable in this environment: {exc}")

    returns, _meta, _sigma = _load("gjrgarch11_norm")
    ours = fit_gjr(returns, cond_dist="norm")
    fitted = arch_model(
        returns * 100.0, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="normal"
    ).fit(disp="off")
    arch_vol = np.asarray(fitted.conditional_volatility) / 100.0
    corr = float(np.corrcoef(ours.conditional_volatility, arch_vol)[0, 1])
    assert corr > 0.999
    assert float(fitted.params["gamma[1]"]) > 0.0  # arch also finds a positive leverage effect
    assert ours.params["gamma1"] > 0.0
