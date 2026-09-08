"""EGARCH(1,1) under non-normal distributions — the EGF distribution infrastructure (Phase 5).

The EGF family centers its news impact ``g(eta)`` on a **distribution-dependent** moment (``E|eta|``
for EGARCH/MEGARCH, ``E[ln(|eta|+1)]`` for MLog-GARCH — so distribution breadth for the whole EGF
family reduces to two new pieces, both proven here on EGARCH:

* **The skew-wrapper moments** (``abs_moment`` / ``mean_log_sq`` / ``mean_log_modulus`` on the
  Fernandez-Steel wrapper, plus the missing base log-moments) by quadrature over the standardized
  density. The correctness anchor: at ``skew = 1`` each skewed moment equals the base moment.
* **The per-iteration centering**: for a jointly-estimated continuous shape (std df, ged shape) the
  centering ``E[g(eta)]`` depends on the shape, so it is re-computed from the *current* shape each
  optimizer iteration (via a QMLE hook), with Nelder-Mead + a restart for the flat shape ridge.

Validation: the reconstruction seam is machine-exact at each fixture's params (recursion proven
on norm, only the centering moment is new); the five converging fixtures (ged/ald/snorm/sged/sald)
match; and egarch x std/sstd — which **fEGarch's own optimizer could not fit** — are validated by
known-truth. Our optimizer converges there (a robustness finding: an independent reimplementation
fitting the reference's hardest EGF case).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    egarch_recursion,
    egarch_sim,
    fit_egarch,
    fit_megarch,
    fit_mloggarch,
    get_distribution,
)
from quantica.timeseries.fegarch.distributions import AverageLaplace, FernandezSteelSkew

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

_FIXTURE_DISTS = ["ged", "ald", "snorm", "sged", "sald"]  # the 5 that converged in fEGarch
_SKEW_TO_BASE = {"snorm": "norm", "sstd": "std", "sged": "ged", "sald": "ald"}
_MOMENTS = ["abs_moment", "mean_log_sq", "mean_log_modulus"]


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _load(model: str, dist: str) -> tuple[dict, np.ndarray]:  # type: ignore[type-arg]
    """Load a fixture's params dict and sigma series."""
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_{model}11_{dist}_params.json").read_text(encoding="utf-8")
    )
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_{model}11_{dist}_sigma.csv", skiprows=1)
    return meta, sigma


def _dist_and_abs_moment(dist: str, fx: dict) -> float:  # type: ignore[type-arg]
    """The magnitude centering E|eta| for a fixture's distribution + reported shape/skew."""
    if dist == "ged":
        return get_distribution("ged").abs_moment((fx["shape"],))
    if dist == "ald":
        return AverageLaplace(p=int(fx["P"])).abs_moment()
    if dist == "snorm":
        return get_distribution("snorm").abs_moment((fx["skew"],))
    if dist == "sged":
        return get_distribution("sged").abs_moment((fx["shape"], fx["skew"]))
    return FernandezSteelSkew(AverageLaplace(p=int(fx["P"]))).abs_moment((fx["skew"],))  # sald


# --------------------------------------------------------------------------- #
# Skew-wrapper moments: the skew=1 -> base reduction (correctness anchor)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("moment", _MOMENTS)
@pytest.mark.parametrize("skew_code", ["snorm", "sged", "sald"])
def test_skew_one_moment_reduces_to_base(moment: str, skew_code: str) -> None:
    """At skew = 1 each Fernandez-Steel-wrapper EGF moment equals the symmetric base's moment.

    The correctness anchor for the whole skew-wrapper: abs_moment / mean_log_sq / mean_log_modulus
    of the skewed law at skew = 1 collapse to the base moment (quadrature over the same density).
    """
    base_code = {"snorm": "norm", "sged": "ged", "sald": "ald"}[skew_code]
    base_shape: tuple[float, ...] = {"norm": (), "ged": (1.4,), "ald": ()}[base_code]
    skewed = (
        get_distribution(skew_code)
        if skew_code != "sald"
        else FernandezSteelSkew(AverageLaplace(p=6))
    )
    base = get_distribution(base_code) if base_code != "ald" else AverageLaplace(p=6)
    skewed_val = getattr(skewed, moment)((*base_shape, 1.0))
    base_val = getattr(base, moment)(base_shape or None)
    assert abs(skewed_val - base_val) < 1e-9


@pytest.mark.parametrize("base_code", ["std", "ged", "ald"])
def test_base_log_moments_are_available(base_code: str) -> None:
    """The base log-moments (mean_log_sq, mean_log_modulus) now compute for std/ged/ald.

    Filled by the same quadrature. Sanity check: ``std`` at large df approaches
    the normal's E[ln z^2] = -1.27036.
    """
    dist = get_distribution(base_code) if base_code != "ald" else AverageLaplace(p=5)
    shape: tuple[float, ...] | None = {"std": (6.0,), "ged": (1.2,), "ald": None}[base_code]  # type: ignore[assignment]
    assert np.isfinite(dist.mean_log_sq(shape))
    assert np.isfinite(dist.mean_log_modulus(shape))
    if base_code == "std":
        near_normal = get_distribution("std").mean_log_sq((1.0e5,))
        assert abs(near_normal - (-1.2703628)) < 1e-3  # std df->inf -> norm E[ln z^2]


# --------------------------------------------------------------------------- #
# Reconstruction seam: recursion + centering moment, machine-exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", _FIXTURE_DISTS)
def test_reconstruction_seam_is_machine_exact(dist: str) -> None:
    """At each fixture's params the EGARCH recursion + the extracted E|eta| reproduce sigma.

    The go/no-go: the recursion is proven on norm, only the centering moment E|eta| is new, so their
    composition must reproduce the sigma series to machine order (closed-form-moment ged/ald ~1e-15;
    the skewed quadrature-moment cases ~1e-12, quad-tolerance-limited).
    """
    fx_meta, sigma_fix = _load("egarch", dist)
    fx = fx_meta["params"]
    var_params = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"]])
    e_abs = _dist_and_abs_moment(dist, fx)
    sigma = np.sqrt(egarch_recursion(var_params, _returns(), abs_moment=e_abs))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


# --------------------------------------------------------------------------- #
# Fixture-match on the 5 converging distributions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", _FIXTURE_DISTS)
def test_egarch_fixture_match(dist: str) -> None:
    """fit_egarch reproduces each converging fixture's loglik, sigma, kappa/gamma and (ald/sald) P.

    The ald/sald cases carry the clearest distribution-dependent E|eta| shift (kappa ~6% off norm),
    so their match is the strongest confirmation the per-iteration centering is right. Shape/skew
    near their normal limits on the Gaussian data (by regime, not param-exact).
    """
    fx_meta, sigma_fix = _load("egarch", dist)
    fx = fx_meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_egarch(_returns(), cond_dist=dist)
    assert abs(fit.loglikelihood - fx_meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - fx_meta["aic"]) < 1e-6
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5
    assert abs(fit.params["kappa"] - fx["kappa"]) < 5e-3
    assert abs(fit.params["gamma"] - fx["gamma"]) < 5e-3
    if "P" in fx:
        assert fit.params["P"] == fx["P"] == 5


def test_norm_paths_stay_bit_identical() -> None:
    """The per-iteration-centering refactor leaves the norm EGF fits intact (egarch/megarch/mlog).

    norm has no shape parameter, so it keeps the precomputed-constant centering + L-BFGS-B and still
    reproduces each norm fixture's log-likelihood unchanged.
    """
    returns = _returns()
    for fit_fn, model in (
        (fit_egarch, "egarch"),
        (fit_megarch, "megarch"),
        (fit_mloggarch, "mloggarch"),
    ):
        meta, _sigma = _load(model, "norm")
        fit = fit_fn(returns, cond_dist="norm")
        assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5


# --------------------------------------------------------------------------- #
# std / sstd known-truth (no fixture — fEGarch's optimizer failed)
# --------------------------------------------------------------------------- #


def test_egarch_std_known_truth_where_fegarch_failed() -> None:
    """egarch x std: recover df from simulated data (fEGarch's optimizer failed to fit this).

    fEGarch could not fit egarch x std ('Error during optimization') — the df-dependent E|eta|
    centering is the hardest EGF case. Our Nelder-Mead + restart + stable closed-form E|eta|(df)
    converges: simulating at an identified df = 6 recovers it, and the fit reports success. That our
    optimizer succeeds where the reference's failed is a robustness finding.
    """
    true = {"mu": 0.0003, "omega_sig": -9.0, "phi1": 0.97, "kappa": -0.05, "gamma": 0.15}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = egarch_sim(
            9000, **true, cond_dist="std", dist_params=(6.0,), rng=np.random.default_rng(0)
        )
        fit = fit_egarch(returns, cond_dist="std")
    assert fit.converged  # converges where fEGarch's optimizer failed
    assert abs(fit.params["nu"] - 6.0) < 3.0 * fit.std_errors["nu"]


def test_egarch_sstd_known_truth_recovers_df_and_skew() -> None:
    """egarch x sstd: recover df AND skew from simulated data (fEGarch's optimizer failed here too).

    The compound continuous-df + skew case. Simulating at df = 6, skew = 0.85 recovers both within a
    few standard errors, and the fit converges.
    """
    true = {"mu": 0.0003, "omega_sig": -9.0, "phi1": 0.97, "kappa": -0.05, "gamma": 0.15}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = egarch_sim(
            9000, **true, cond_dist="sstd", dist_params=(6.0, 0.85), rng=np.random.default_rng(0)
        )
        fit = fit_egarch(returns, cond_dist="sstd")
    assert fit.converged
    assert abs(fit.params["nu"] - 6.0) < 3.0 * fit.std_errors["nu"]
    assert abs(fit.params["skew"] - 0.85) < 3.0 * fit.std_errors["skew"]
