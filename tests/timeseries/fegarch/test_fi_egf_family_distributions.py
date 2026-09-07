"""FIEGARCH / FIMEGARCH / FIMLog-GARCH / FILog-GARCH under non-normal distributions (Phase 5).

The **long-memory EGF family** under the eight distributions — the last layer of the distribution
breadth, completing every fEGarch model under all eight laws. This is **true inheritance**: the FI
models reuse, unchanged, the four EGF centering moments (``abs_moment`` / ``mean_log_sq`` /
``mean_log_modulus`` / ``mean_signed_log_modulus``) and the per-iteration centering proven on the
short-memory EGF family, spliced into the truncated-MA(infinity) ``theta(B)`` recursion. No new
distribution mathematics — the reconstruction seam is machine-exact for all 22 converging fixtures
(worst ``~2e-10``) purely by composition.

Validation is **seam-centric** (the §14/§19 effective-challenge discipline): the machine-exact seam
is the correctness proof; the *fits* are weakly identified on the flat ``d`` + shape/skew ridge, so
they **dominate-or-tie** the fixtures rather than matching them. A tight param-match is asserted
**only** where the fixture is genuinely well-identified (``|loglik dev| < 1e-4``); everywhere else
(fiegarch/fimegarch sged/sald, fimloggarch ald/sald, all of FILog-GARCH's near-common-root ridge,
and FILog-GARCH's strictly-dominated norm fixture at ``+120``) the assertion is dominate-or-tie.

**The P-profiling discriminating test.** FILog-GARCH x sald is the one fixture where fEGarch
selected an *interior* ``P = 3`` rather than the boundary. Verified read-only: our per-``P``
optimizer reproduces fEGarch's ``P = 3`` point exactly (``7550.13``) but fits ``P = 4`` and
``P = 5`` to genuine higher converged optima fEGarch missed — a smooth monotone-concave profile
peaking at ``P = 5`` (``7556.80``). So our profiling **selects ``P = 5`` and dominates** the
fixture's ``P = 3``: fEGarch's lone interior-P was an **optimizer stall on the flat ridge**, the §14
strictly-dominated pattern now on the discrete P-axis — proof our P-profiling *compares* per-P
likelihoods (it lands fEGarch's value at each P) rather than defaulting to the boundary.

**The Type-I-fails / Type-II-converges split.** fEGarch's optimizer failed the continuous-df fits
for all three Type-I FI models (fiegarch/fimegarch/fimloggarch x std/sstd, 6 cases) but converged
for the Type-II FILog-GARCH on the same std/sstd — so FILog-GARCH has all 7 non-norm fixtures while
the Type-I trio have 5 each (22 total). The 6 Type-I failures are validated by known-truth: our
optimizer converges on all 6.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fiegarch_recursion,
    fiegarch_sim,
    filoggarch_recursion,
    fimegarch_recursion,
    fimegarch_sim,
    fimloggarch_recursion,
    fimloggarch_sim,
    fit_fiegarch,
    fit_filoggarch,
    fit_fimegarch,
    fit_fimloggarch,
    get_distribution,
)
from quantica.timeseries.fegarch.distributions import AverageLaplace, FernandezSteelSkew

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# fEGarch converged for the Type-I trio only on the 5 non-continuous-df laws (std/sstd failed);
# for the Type-II FILog-GARCH it converged on all 7 -- the Type-I-fails/Type-II-converges split.
_TYPE1_FI_DISTS = ["ged", "ald", "snorm", "sged", "sald"]
_FILOGGARCH_DISTS = ["std", "ged", "ald", "snorm", "sstd", "sged", "sald"]
_TYPE1_FIT = {"fiegarch": fit_fiegarch, "fimegarch": fit_fimegarch, "fimloggarch": fit_fimloggarch}

# The tight/dominate split is per-(model, dist) and read from the committed fixtures (spec-notes
# §22). TIGHT = the fixture is well-identified, |loglik dev| < 1e-4 -> assert a param match. Every
# other converging case is weakly identified on the flat d + shape/skew ridge -> dominate-or-tie.
_TIGHT: set[tuple[str, str]] = {
    ("fiegarch", "ged"), ("fiegarch", "ald"), ("fiegarch", "snorm"),
    ("fimegarch", "ged"), ("fimegarch", "ald"), ("fimegarch", "snorm"),
    ("fimloggarch", "ged"), ("fimloggarch", "snorm"), ("fimloggarch", "sged"),
}  # fmt: skip
_TYPE1_DOMINATES = [  # the Type-I non-tight cases -- fit reaches-or-beats the fixture
    (m, d) for m in _TYPE1_FIT for d in _TYPE1_FI_DISTS if (m, d) not in _TIGHT
]

# Per-push CI runs one fast representative of each fit-category (plus all seams, the gamma tell, and
# the norm fits); the exhaustive fit sweep AND the P-profiling headline are marked ``slow`` and run
# in the opt-in/scheduled tier -- each FI fit is O(n^2) (the coupled truncated-MA(inf) recursion),
# each ald/sald fit 5x that via P-profiling (the sald profile alone is ~20-40min), so the full sweep
# is ~2-3h. The P-profiling finding is also documented in spec-notes §22.4 + the read-only check.
_TIGHT_FAST = ("fiegarch", "ged")  # a well-identified continuous-shape representative
_DOMINATE_FAST = ("fiegarch", "sged")  # a dominating continuous-shape representative (+3.8)
_TIGHT_CASES = [
    _TIGHT_FAST,
    *(
        pytest.param(m, d, marks=pytest.mark.slow)
        for m, d in sorted(_TIGHT)
        if (m, d) != _TIGHT_FAST
    ),
]
_DOMINATE_CASES = [
    _DOMINATE_FAST,
    *(
        pytest.param(m, d, marks=pytest.mark.slow)
        for m, d in _TYPE1_DOMINATES
        if (m, d) != _DOMINATE_FAST
    ),
]
# FILog-GARCH's ridge domination is represented in per-push by the P-profiling headline (sald, +6.7)
# and test_filoggarch_norm_dominates (+120); the full 7-dist ridge sweep is slow.
_FILOGGARCH_CASES = [pytest.param(d, marks=pytest.mark.slow) for d in _FILOGGARCH_DISTS]
# FILog-GARCH's near-common-root ridge: 6 of 7 strictly dominate (+6..+45); ald ties within ~0.8.
_FILOG_DOMINATES = {"std", "ged", "snorm", "sstd", "sged", "sald"}


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


def _dist_and_params(dist: str, fx: dict) -> tuple[object, tuple[float, ...] | None]:  # type: ignore[type-arg]
    """Distribution instance + shape params for a fixture's reported shape/skew."""
    if dist == "std":
        return get_distribution("std"), (fx["df"],)
    if dist == "ged":
        return get_distribution("ged"), (fx["shape"],)
    if dist == "ald":
        return AverageLaplace(p=int(fx["P"])), None
    if dist == "snorm":
        return get_distribution("snorm"), (fx["skew"],)
    if dist == "sstd":
        return get_distribution("sstd"), (fx["df"], fx["skew"])
    if dist == "sged":
        return get_distribution("sged"), (fx["shape"], fx["skew"])
    return FernandezSteelSkew(AverageLaplace(p=int(fx["P"]))), (fx["skew"],)  # sald


# --------------------------------------------------------------------------- #
# Reconstruction seam: recursion + centering moments, machine-exact (all 22)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", _TYPE1_FI_DISTS)
def test_fiegarch_seam_is_machine_exact(dist: str) -> None:
    """FIEGARCH seam: the E|eta| magnitude reproduces sigma (asymmetry g_asy=eta centers to 0)."""
    fx_meta, sigma_fix = _load("fiegarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"], fx["d"]])
    sigma = np.sqrt(fiegarch_recursion(vp, _returns(), abs_moment=d.abs_moment(p)))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


@pytest.mark.parametrize("dist", _TYPE1_FI_DISTS)
def test_fimegarch_seam_is_machine_exact(dist: str) -> None:
    """FIMEGARCH seam: E|eta| magnitude + the signed-log-modulus asymmetry reproduce sigma.

    The skewed cases exercise the 4th EGF moment through the long-memory theta(B) recursion -- the
    same asymmetry centering proven on short-memory MEGARCH, inherited unchanged.
    """
    fx_meta, sigma_fix = _load("fimegarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"], fx["d"]])
    sigma = np.sqrt(
        fimegarch_recursion(
            vp,
            _returns(),
            abs_moment=d.abs_moment(p),
            mean_signed_log_modulus=d.mean_signed_log_modulus(p),
        )
    )
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


@pytest.mark.parametrize("dist", _TYPE1_FI_DISTS)
def test_fimloggarch_seam_is_machine_exact(dist: str) -> None:
    """FIMLog-GARCH seam: E[ln(|eta|+1)] magnitude + signed-log-modulus asymmetry rebuild sigma."""
    fx_meta, sigma_fix = _load("fimloggarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"], fx["d"]])
    sigma = np.sqrt(
        fimloggarch_recursion(
            vp,
            _returns(),
            mean_log_modulus=d.mean_log_modulus(p),
            mean_signed_log_modulus=d.mean_signed_log_modulus(p),
        )
    )
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


@pytest.mark.parametrize("dist", _FILOGGARCH_DISTS)
def test_filoggarch_seam_is_machine_exact(dist: str) -> None:
    """FILog-GARCH seam: the E[ln eta^2] centering reproduces sigma (Type-II, no asymmetry).

    All 7 non-norm laws converged in fEGarch for the Type-II model (std/sstd included) -- the
    Type-I-fails/Type-II-converges split -- so the seam is proven on the full distribution set.
    """
    fx_meta, sigma_fix = _load("filoggarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["psi1"], fx["d"]])
    sigma = np.sqrt(filoggarch_recursion(vp, _returns(), mean_log_sq=d.mean_log_sq(p)))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


# --------------------------------------------------------------------------- #
# Fixture-match: tight only where well-identified; else dominate-or-tie
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("model", "dist"), _TIGHT_CASES)
def test_type1_fi_fixture_match_tight(model: str, dist: str) -> None:
    """The 9 well-identified Type-I FI fixtures (|loglik dev| < 1e-4) match tightly.

    These are the sharply-identified cases; the fit reproduces the fixture's loglik, sigma, gamma
    and (ald) P. The dominated cases (sged/sald, fimloggarch ald) are handled separately -- no
    param-match against a weakly-identified fixture.
    """
    fx_meta, sigma_fix = _load(model, dist)
    fx = fx_meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _TYPE1_FIT[model](_returns(), cond_dist=dist)
    assert fit.converged
    assert abs(fit.loglikelihood - fx_meta["loglikelihood"]) < 1e-4
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5
    assert abs(fit.params["gamma"] - fx["gamma"]) < 5e-3
    if "P" in fx:
        assert fit.params["P"] == fx["P"] == 5


@pytest.mark.parametrize(("model", "dist"), _DOMINATE_CASES)
def test_type1_fi_fixture_dominates_or_ties(model: str, dist: str) -> None:
    """The Type-I FI cases on the flat d + shape/skew ridge reach-or-beat the fixture.

    Weakly identified, so our optimizer finds a comparable-or-higher-likelihood point than fEGarch
    (fiegarch/fimegarch sged +3.8, fimloggarch sald +4.1 -- strictly dominating). The seam is
    machine-exact (correctness proven above); validated by domination, not a param match, so the
    profiled P may differ (a dominated-ridge P-shift) and is not asserted.
    """
    fx_meta, _sigma = _load(model, dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _TYPE1_FIT[model](_returns(), cond_dist=dist)
    assert fit.converged
    assert fit.loglikelihood >= fx_meta["loglikelihood"] - 1e-3


@pytest.mark.parametrize("dist", _FILOGGARCH_CASES)
def test_filoggarch_fixture_dominates_or_ties(dist: str) -> None:
    """FILog-GARCH sits on the near-common-root ridge -- the fit dominates-or-ties every fixture.

    6 of 7 strictly dominate (+6 to +45 loglik; the §14 strictly-dominated-fixture pattern under
    distributions); ald lands ~0.76 below on the ridge (comparable, not a failure). The seam is
    machine-exact, so this is fEGarch's optimizer on the ridge, not the model -- validated by
    domination/comparability, never a param match.
    """
    fx_meta, _sigma = _load("filoggarch", dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_filoggarch(_returns(), cond_dist=dist)
    assert fit.converged
    if dist in _FILOG_DOMINATES:
        assert fit.loglikelihood > fx_meta["loglikelihood"] + 3.0  # strictly dominates the ridge
    else:  # ald -- comparable on the ridge (our fit 0.76 below fEGarch's ridge point)
        assert fit.loglikelihood >= fx_meta["loglikelihood"] - 1.0


def test_filoggarch_norm_dominates_the_dominated_fixture() -> None:
    """FILog-GARCH's norm fixture is a strictly-dominated local optimum (mu=-0.003) -- §14.

    Every data-driven start escapes to a basin ~+120 loglik higher, so the norm fit dominates the
    committed fixture by ~120 (not a regression; the fixture is fEGarch's poor local optimum).
    """
    fx_meta, _sigma = _load("filoggarch", "norm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # benign overflow-in-exp while the optimizer explores
        fit = fit_filoggarch(_returns(), cond_dist="norm")
    assert fit.converged
    assert fit.loglikelihood > fx_meta["loglikelihood"] + 100.0


# --------------------------------------------------------------------------- #
# The P-profiling discriminating test: FILog-GARCH x sald (P=3 is fEGarch's stall)
# --------------------------------------------------------------------------- #


@pytest.mark.slow  # a full 5-P ALD profile of the flat-ridge sald fit -- the single costliest test
def test_filoggarch_sald_p_profiling_selects_true_max_and_dominates_fegarchs_interior_p() -> None:
    """FILog-GARCH x sald: our profiling selects P=5 (the true max) and dominates fEGarch's P=3.

    fEGarch's fixture selected an *interior* P=3 -- the only interior-P selection in the whole port.
    Verified read-only that P=4/P=5 are clean converged optima (d interior, in-region, valid sigma):
    (1) our profiling selects P=5, the true monotone-concave maximum;
    (2) at P=3 our fit reproduces fEGarch's P=3 loglik (7550.13) to ~1e-3 -- we *can* land fEGarch's
        point, so this is not a different objective;
    (3) it therefore dominates the fixture's P=3 selection (7556.80 > 7550.13),
    documenting fEGarch's interior P=3 as an optimizer stall on the flat ridge (the §14
    strictly-dominated pattern on the discrete P-axis), NOT a real interior maximum. Proof the
    P-profiling *compares* per-P likelihoods rather than defaulting to the boundary.
    """
    fx_meta, _sigma = _load("filoggarch", "sald")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_filoggarch(_returns(), cond_dist="sald")
    assert fit.converged
    # fEGarch's fixture selected the interior P=3; our profiling selects the boundary P=5.
    assert fx_meta["params"]["P"] == 3
    assert fit.params["P"] == 5
    profile = dict(fit.profile)  # {P: loglik}
    assert set(profile) == {1, 2, 3, 4, 5}
    # (2) at P=3 we reproduce fEGarch's P=3 loglik -- we land their point.
    assert abs(profile[3] - fx_meta["loglikelihood"]) < 1e-2
    # (1) the profile is smooth monotone-concave -> a genuine boundary maximum, not a degeneracy.
    lls = [profile[p] for p in (1, 2, 3, 4, 5)]
    diffs = [lls[i + 1] - lls[i] for i in range(4)]
    assert all(step > 0 for step in diffs)  # strictly increasing -> max at P=5
    assert all(diffs[i + 1] < diffs[i] for i in range(3))  # decelerating (concave), no jump
    # (3) the selected P=5 dominates fEGarch's interior P=3 (7556.80 vs 7550.13).
    assert fit.loglikelihood > fx_meta["loglikelihood"] + 5.0


# --------------------------------------------------------------------------- #
# The gamma tell + norm bit-identical
# --------------------------------------------------------------------------- #


def test_gamma_tell_fimloggarch_is_1p85x_fimegarch() -> None:
    """The log-modulus magnitude tell carries into long memory: FIMLog-GARCH gamma ~ 1.85x FIMEG.

    FIMEGARCH's |eta| magnitude gives gamma ~ 0.17; FIMLog-GARCH's compressed ln(|eta|+1) gives
    ~0.32, a ~1.85x ratio held across every converging law (read from the committed fixtures).
    """
    for dist in ["norm", *_TYPE1_FI_DISTS]:
        g_me = _load("fimegarch", dist)[0]["params"]["gamma"]
        g_ml = _load("fimloggarch", dist)[0]["params"]["gamma"]
        assert 1.80 < g_ml / g_me < 1.90


def test_fi_norm_paths_stay_bit_identical() -> None:
    """The per-iteration + asymmetry-centering refactor leaves the three Type-I FI norm fits intact.

    norm has no shape parameter and a zero asymmetry centering, so each keeps its precomputed
    L-BFGS-B constant path and reproduces its norm fixture log-likelihood unchanged (FILog-GARCH
    norm is the strictly-dominated-fixture case, asserted separately).
    """
    returns = _returns()
    for model, fit_fn in _TYPE1_FIT.items():
        meta, _sigma = _load(model, "norm")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # benign overflow-in-exp while the optimizer explores
            fit = fit_fn(returns, cond_dist="norm")
        assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5


# --------------------------------------------------------------------------- #
# Known-truth for the 6 Type-I continuous-df fEGarch-optimizer failures
# --------------------------------------------------------------------------- #


# One combo (fiegarch x std) runs per-push; the other 5 are slow (each is a continuous-shape fit on
# an n=3000 O(n^2) FI recursion). All 6 were verified to converge and recover df/skew within 3*SE.
_KNOWN_TRUTH_CASES = [
    (fiegarch_sim, fit_fiegarch, 0.15, "std"),  # fast representative
    pytest.param(fiegarch_sim, fit_fiegarch, 0.15, "sstd", marks=pytest.mark.slow),
    pytest.param(fimegarch_sim, fit_fimegarch, 0.15, "std", marks=pytest.mark.slow),
    pytest.param(fimegarch_sim, fit_fimegarch, 0.15, "sstd", marks=pytest.mark.slow),
    pytest.param(fimloggarch_sim, fit_fimloggarch, 0.25, "std", marks=pytest.mark.slow),
    pytest.param(fimloggarch_sim, fit_fimloggarch, 0.25, "sstd", marks=pytest.mark.slow),
]


@pytest.mark.parametrize(("sim", "fit_fn", "gamma", "dist"), _KNOWN_TRUTH_CASES)
def test_type1_fi_known_truth_where_fegarch_failed(sim, fit_fn, gamma, dist) -> None:  # type: ignore[no-untyped-def]
    """fiegarch/fimegarch/fimloggarch x std|sstd: recover df (+skew) where fEGarch's optimizer died.

    fEGarch failed all three Type-I FI models on continuous df (std/sstd) while the Type-II
    FILog-GARCH converged on the same laws -- the Type-I-fails/Type-II-converges split. No fixture
    exists (none fabricated); validated by known-truth on a long-memory (d=0.3) simulation. The sstd
    case exercises the per-iteration asymmetry centering (nonzero under skew) in both sim and fit.
    """
    dist_params = (6.0,) if dist == "std" else (6.0, 0.85)
    true = {"mu": 0.0003, "omega_sig": -9.0, "phi1": 0.4, "d": 0.3, "kappa": -0.05, "gamma": gamma}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # n = 3000 (not the 9000 of the O(n) short-memory known-truth): the FI truncated-MA(inf)
        # recursion is O(n^2), so six continuous-shape fits at 9000 would dominate suite runtime.
        returns, _sigma = sim(
            3000, **true, cond_dist=dist, dist_params=dist_params, rng=np.random.default_rng(0)
        )
        fit = fit_fn(returns, cond_dist=dist)
    assert fit.converged
    assert abs(fit.params["nu"] - 6.0) < 3.0 * fit.std_errors["nu"]
    if dist == "sstd":
        assert abs(fit.params["skew"] - 0.85) < 3.0 * fit.std_errors["skew"]
