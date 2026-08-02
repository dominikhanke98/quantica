"""Validation of the fEGarch conditional-distribution layer (numerical-validation skill).

These are the checks that hold **without** fEGarch fixtures: each standardized density integrates
to 1 with mean 0 and variance 1; ``cdf`` and ``ppf`` are mutual inverses; the seeded sampler's
empirical moments match the analytic ones and are deterministic; and each Fernández-Steel skew
reduces exactly to its symmetric base at :math:`\\xi = 1` (the clean known-truth anchor). The
"matches fEGarch to tolerance" checks are deferred to explicit skipped stubs below until the R
output fixtures are committed (CLAUDE.md §12) — so nothing here silently claims agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.timeseries.fegarch import DISTRIBUTIONS, get_distribution
from quantica.timeseries.fegarch.distributions import FernandezSteelSkew
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
    draws = dist.sample(200_000, np.random.default_rng(0), params)
    again = dist.sample(200_000, np.random.default_rng(0), params)
    assert np.array_equal(draws, again)  # deterministic given the seed
    assert abs(float(draws.mean())) < 0.02  # analytic mean 0
    assert abs(float(draws.var()) - 1.0) < 0.03  # analytic variance 1


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
# Deferred: agreement with fEGarch itself (needs committed R output fixtures)
# --------------------------------------------------------------------------- #

_FIXTURE_REASON = (
    "fEGarch output fixtures not yet committed — Phase 0 follow-up (CLAUDE.md §12: validate "
    "against committed fEGarch output, generated once in R). RECONCILE the ALD form and the "
    "Fernández-Steel normalization constants here once the fixtures exist."
)


@pytest.mark.skip(reason=_FIXTURE_REASON)
@pytest.mark.parametrize("code", _ALL)
def test_density_matches_fegarch_fixture(code: str) -> None:
    """Each standardized density matches fEGarch's density values on committed fixture points."""
    raise AssertionError("fixture not available yet")  # pragma: no cover


@pytest.mark.skip(reason=_FIXTURE_REASON)
def test_ald_form_matches_fegarch_fixture() -> None:
    """RECONCILE: the exact average-Laplace density matches fEGarch (locks the ALD default)."""
    raise AssertionError("fixture not available yet")  # pragma: no cover


@pytest.mark.skip(reason=_FIXTURE_REASON)
def test_fernandez_steel_constants_match_fegarch_fixture() -> None:
    """RECONCILE: the skew normalization constants match fEGarch (locks the FS parameterization)."""
    raise AssertionError("fixture not available yet")  # pragma: no cover
