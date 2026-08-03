"""Validation of the fEGarch conditional-distribution layer (numerical-validation skill).

Two tiers of checks. **Analytic** (fixture-free): each standardized density integrates to 1 with
mean 0 and variance 1; ``cdf`` and ``ppf`` are mutual inverses; the seeded sampler's empirical
moments match the analytic ones and are deterministic; each Fernández-Steel skew reduces exactly to
its symmetric base at :math:`\\xi = 1`; and the average-Laplace moment identities
(``a(0)=a(2)=1``, ``a(4)=3+3/(P+1)``) hold. **Against fEGarch** (CLAUDE.md §12): the committed R
output fixtures are fEGarch's *sampler* output (empirical, 1e6 draws — fEGarch exposes no public
density/quantile functions), so the analytic quantiles / moments are matched within Monte-Carlo
tolerance (~0.02 on a unit-sd quantile), not to machine precision. These fixture tests lock the ALD
form (scaled average-Laplace, not Laplace) and the Fernández-Steel skew convention (``skew = xi``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import DISTRIBUTIONS, get_distribution
from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    FernandezSteelSkew,
    GeneralizedError,
    Normal,
    StudentT,
)
from scipy import integrate

_ALL = sorted(DISTRIBUTIONS)
_SYMMETRIC_TO_SKEW = [("norm", "snorm"), ("std", "sstd"), ("ged", "sged"), ("ald", "sald")]

# Non-trivial parameter sets (heavier tails / genuine skew) to exercise beyond the defaults.
_STRESS_PARAMS = {
    "norm": (),
    "std": (5.0,),
    "ged": (1.2,),
    "ald": (),
    "snorm": (1.5,),
    "sstd": (6.0, 0.8),
    "sged": (1.3, 1.3),
    "sald": (0.7,),
}


@pytest.mark.parametrize("code", _ALL)
def test_density_is_a_standardized_distribution(code: str) -> None:
    """Each pdf integrates to 1 with mean 0 and variance 1 (the QMLE standardization)."""
    dist = get_distribution(code)
    params = _STRESS_PARAMS[code]

    def pdf(z: float) -> float:
        return float(dist.pdf(np.array([z]), params)[0])

    # Infinite limits so the heavy-tailed densities' variance integrates accurately.
    total = integrate.quad(pdf, -np.inf, np.inf, limit=200)[0]
    mean = integrate.quad(lambda z: z * pdf(z), -np.inf, np.inf, limit=200)[0]
    second = integrate.quad(lambda z: z * z * pdf(z), -np.inf, np.inf, limit=200)[0]
    assert abs(total - 1.0) < 1e-6
    assert abs(mean) < 1e-5
    assert abs(second - 1.0) < 1e-4  # variance = E[z^2] since mean 0


@pytest.mark.parametrize("code", _ALL)
def test_pdf_matches_exp_logpdf(code: str) -> None:
    """``pdf`` equals ``exp(logpdf)`` elementwise."""
    dist = get_distribution(code)
    params = _STRESS_PARAMS[code]
    z = np.array([-2.3, -0.5, 0.0, 0.8, 3.1])
    assert np.allclose(dist.pdf(z, params), np.exp(dist.logpdf(z, params)))


@pytest.mark.parametrize("code", _ALL)
def test_cdf_ppf_round_trip(code: str) -> None:
    """``ppf(cdf(z)) = z`` and ``cdf(ppf(p)) = p`` to tolerance."""
    dist = get_distribution(code)
    params = _STRESS_PARAMS[code]
    z = np.array([-3.0, -1.1, -0.2, 0.4, 1.6, 2.7])
    assert np.allclose(dist.ppf(dist.cdf(z, params), params), z, atol=1e-8)
    p = np.array([0.02, 0.17, 0.4, 0.63, 0.85, 0.98])
    assert np.allclose(dist.cdf(dist.ppf(p, params), params), p, atol=1e-8)


@pytest.mark.parametrize("code", _ALL)
def test_cdf_is_monotone_and_bounded(code: str) -> None:
    """The CDF is in [0, 1] and non-decreasing across a grid."""
    dist = get_distribution(code)
    params = _STRESS_PARAMS[code]
    grid = np.linspace(-6.0, 6.0, 200)
    cdf = dist.cdf(grid, params)
    assert np.all(cdf >= -1e-9) and np.all(cdf <= 1.0 + 1e-9)
    assert np.all(np.diff(cdf) >= -1e-12)


@pytest.mark.parametrize("code", _ALL)
def test_sampler_moments_and_determinism(code: str) -> None:
    """The seeded sampler is deterministic and its empirical moments match the analytic ones."""
    dist = get_distribution(code)
    params = _STRESS_PARAMS[code]
    # 50k keeps the moment SE tight while the average-Laplace numeric ppf stays fast.
    draws = dist.sample(50_000, np.random.default_rng(0), params)
    again = dist.sample(50_000, np.random.default_rng(0), params)
    assert np.array_equal(draws, again)  # deterministic given the seed
    assert abs(float(draws.mean())) < 0.03  # analytic mean 0
    assert abs(float(draws.var()) - 1.0) < 0.05  # analytic variance 1


@pytest.mark.parametrize(("base_code", "skew_code"), _SYMMETRIC_TO_SKEW)
def test_skew_reduces_to_base_at_unit_skew(base_code: str, skew_code: str) -> None:
    """At :math:`\\xi = 1` the skew density/CDF/quantile equal the symmetric base (known truth)."""
    base = get_distribution(base_code)
    skew = get_distribution(skew_code)
    base_params = base.param_start
    skew_params = (*base_params, 1.0)
    z = np.array([-2.0, -0.7, 0.3, 1.4, 2.9])
    p = np.array([0.05, 0.3, 0.5, 0.72, 0.95])
    assert np.allclose(skew.logpdf(z, skew_params), base.logpdf(z, base_params), atol=1e-10)
    assert np.allclose(skew.cdf(z, skew_params), base.cdf(z, base_params), atol=1e-10)
    assert np.allclose(skew.ppf(p, skew_params), base.ppf(p, base_params), atol=1e-8)


def test_skew_actually_skews() -> None:
    """A skew parameter away from 1 produces genuine asymmetry (nonzero third moment)."""
    skew = get_distribution("snorm")

    def pdf(z: float) -> float:
        return float(skew.pdf(np.array([z]), (1.6,))[0])

    third = integrate.quad(lambda z: z**3 * pdf(z), -40, 40)[0]
    assert abs(third) > 0.1  # clearly skewed, unlike the symmetric base (third moment 0)


@pytest.mark.parametrize("p_order", [1, 2, 8])
def test_average_laplace_moment_identities(p_order: int) -> None:
    """The scaled average-Laplace absolute moments satisfy a(0)=a(2)=1, a(4)=3+3/(P+1); F(0)=0.5."""
    ald = AverageLaplace(p_order)
    assert np.isclose(ald.absolute_moment(0), 1.0)  # normalized
    assert np.isclose(ald.absolute_moment(2), 1.0)  # unit variance
    assert np.isclose(ald.absolute_moment(4), 3.0 + 3.0 / (p_order + 1))  # raw kurtosis identity
    assert np.isclose(float(ald.cdf(np.array([0.0]))[0]), 0.5)
    # numeric-inversion ppf round-trips its own cdf
    z = np.array([-2.4, -0.6, 0.9, 3.2])
    assert np.allclose(ald.ppf(ald.cdf(z)), z, atol=1e-8)


def test_average_laplace_rejects_bad_order() -> None:
    """The construction parameter P must be an integer >= 1."""
    with pytest.raises(ValueError, match="P must be an integer >= 1"):
        AverageLaplace(0)


@pytest.mark.parametrize("base", [Normal(), StudentT(), GeneralizedError(), AverageLaplace(8)])
def test_fernandez_steel_unit_skew_is_neutral(base: object) -> None:
    """At xi=1 the FS standardizing constants are C_E=0 (mean) and C_V=1 (sd) for every base."""
    skew = FernandezSteelSkew(base)  # type: ignore[arg-type]
    mu, sigma = skew._mu_sigma(base.param_start, 1.0)  # type: ignore[attr-defined]
    assert np.isclose(mu, 0.0)
    assert np.isclose(sigma, 1.0)


def test_unknown_distribution_raises() -> None:
    """Looking up an unsupported code raises a clear error."""
    with pytest.raises(KeyError, match="unknown distribution"):
        get_distribution("laplace")


def test_wrong_parameter_count_raises() -> None:
    """Passing the wrong number of shape parameters is rejected."""
    with pytest.raises(ValueError, match="expects 1 params"):
        get_distribution("std").logpdf(np.array([0.0]), ())
    with pytest.raises(ValueError, match="expects 2 params"):
        get_distribution("sstd").logpdf(np.array([0.0]), (5.0,))


def test_registry_covers_the_eight_fegarch_codes() -> None:
    """The registry is exactly the eight fEGarch distribution codes, four of them skewed."""
    assert set(DISTRIBUTIONS) == {"norm", "std", "ged", "ald", "snorm", "sstd", "sged", "sald"}
    skewed = [c for c, d in DISTRIBUTIONS.items() if isinstance(d, FernandezSteelSkew)]
    assert sorted(skewed) == ["sald", "sged", "snorm", "sstd"]


# --------------------------------------------------------------------------- #
# Agreement with fEGarch — committed R output fixtures (CLAUDE.md §12)
#
# The fixtures are fEGarch's *sampler* output (empirical, 1e6 draws; fEGarch exposes no public
# density/quantile functions), so analytic values are matched within Monte-Carlo tolerance, not to
# machine precision. See tests/fixtures/fegarch/manifest.json for provenance.
# --------------------------------------------------------------------------- #

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_QUANTILE_ATOL = 0.02  # ~a few Monte-Carlo SEs on a unit-sd quantile (1e6 draws)


def _parse_args(args: str) -> dict[str, str]:
    """Parse a fixture ``args`` cell like ``df=6;skew=0.85`` into a dict."""
    return dict(kv.split("=") for kv in args.split(";")) if args else {}


def _build_distribution(dist: str, args: str) -> tuple[object, tuple[float, ...]]:
    """Rebuild the distribution + params for a fixture row (P/df/shape/skew read from ``args``)."""
    a = _parse_args(args)
    if dist == "norm":
        return Normal(), ()
    if dist == "std":
        return StudentT(), (float(a["df"]),)
    if dist == "ged":
        return GeneralizedError(), (float(a["shape"]),)
    if dist == "ald":
        return AverageLaplace(int(a["P"])), ()
    if dist == "snorm":
        return FernandezSteelSkew(Normal()), (float(a["skew"]),)
    if dist == "sstd":
        return FernandezSteelSkew(StudentT()), (float(a["df"]), float(a["skew"]))
    if dist == "sged":
        return FernandezSteelSkew(GeneralizedError()), (float(a["shape"]), float(a["skew"]))
    if dist == "sald":
        return FernandezSteelSkew(AverageLaplace(int(a["P"]))), (float(a["skew"]),)
    raise ValueError(f"unknown fixture dist {dist!r}")


def _load_moments() -> dict[str, dict[str, str]]:
    """Load ``distribution_moments.csv`` keyed by label."""
    with (_FIXTURE_DIR / "distribution_moments.csv").open(encoding="utf-8") as handle:
        return {row["label"]: row for row in csv.DictReader(handle)}


def _load_quantiles() -> dict[str, tuple[np.ndarray, np.ndarray]]:  # type: ignore[type-arg]
    """Load ``distribution_quantiles.csv`` keyed by label -> (probs, quantiles)."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    with (_FIXTURE_DIR / "distribution_quantiles.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["label"], []).append(
                (float(row["prob"]), float(row["quantile"]))
            )
    return {
        label: (np.array([p for p, _ in pairs]), np.array([q for _, q in pairs]))
        for label, pairs in grouped.items()
    }


_MOMENTS = _load_moments()
_QUANTILES = _load_quantiles()
_FIXTURE_LABELS = sorted(_MOMENTS)


def _analytic_skewness(dist: object, params: tuple[float, ...]) -> float:
    """The analytic skewness E[z^3] of a standardized (mean-0, var-1) distribution."""

    def pdf(z: float) -> float:
        return float(dist.pdf(np.array([z]), params)[0])  # type: ignore[attr-defined]

    return integrate.quad(lambda z: z**3 * pdf(z), -40, 40)[0]


@pytest.mark.parametrize("label", _FIXTURE_LABELS)
def test_density_matches_fegarch_fixture(label: str) -> None:
    """Analytic quantiles match fEGarch's, and fEGarch's output is standardized like ours."""
    row = _MOMENTS[label]
    probs, fixture_q = _QUANTILES[label]
    dist, params = _build_distribution(row["dist"], row["args"])
    # Quantiles: analytic ppf vs fEGarch's empirical quantiles (Monte-Carlo tolerance).
    assert np.allclose(dist.ppf(probs, params), fixture_q, atol=_QUANTILE_ATOL)  # type: ignore[attr-defined]
    # Standardization: fEGarch's sample is mean-0 / variance-1, confirming our QMLE convention.
    assert abs(float(row["mean"])) < 0.01
    assert abs(float(row["variance"]) - 1.0) < 0.01


def test_ald_form_matches_fegarch_fixture() -> None:
    """RECONCILE locked: fEGarch ``ald`` is the scaled average-Laplace (Sargan), not the Laplace."""
    for label, p_order in [("ald_P2", 2), ("ald_P8", 8)]:
        row = _MOMENTS[label]
        probs, fixture_q = _QUANTILES[label]
        ald = AverageLaplace(p_order)
        # The exact kurtosis identity matches fEGarch's empirical kurtosis (a standardized Laplace
        # would give 6 — confirm fEGarch is the lighter-tailed Sargan form).
        assert np.isclose(ald.absolute_moment(4), 3.0 + 3.0 / (p_order + 1))
        assert abs(ald.absolute_moment(4) - float(row["kurtosis_raw"])) < 0.05
        assert float(row["kurtosis_raw"]) < 4.5  # decisively not the Laplace (kurtosis 6)
        assert np.allclose(ald.ppf(probs), fixture_q, atol=_QUANTILE_ATOL)


def test_fernandez_steel_constants_match_fegarch_fixture() -> None:
    """RECONCILE locked: the FS normalization matches fEGarch and ``skew = xi`` directly."""
    skewed = [
        "snorm_skew0.8",
        "snorm_skew1.3",
        "sstd_df6_skew0.85",
        "sged_shape1.3_skew1.2",
        "sald_P8_skew0.8",
    ]
    for label in skewed:
        row = _MOMENTS[label]
        probs, fixture_q = _QUANTILES[label]
        dist, params = _build_distribution(row["dist"], row["args"])
        skew = float(_parse_args(row["args"])["skew"])
        # Analytic skewness matches fEGarch's empirical (confirms the standardization constants).
        assert abs(_analytic_skewness(dist, params) - float(row["skewness"])) < 0.05
        # Direct convention: skew < 1 -> left-skew (negative), skew > 1 -> right-skew (positive).
        assert np.sign(float(row["skewness"])) == np.sign(skew - 1.0)
        # And the quantiles line up.
        assert np.allclose(dist.ppf(probs, params), fixture_q, atol=_QUANTILE_ATOL)  # type: ignore[attr-defined]
