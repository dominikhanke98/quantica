"""Phase 7 sub-build 1 -- local-polynomial regression (the semiparametric scale core).

Validated against the committed smoots::gsmooth fixtures
(``smooth_gsmooth_p{1,3}_mu{0,1,2,3}_bb{0,1}``) on the log-squared demeaned returns w_t = log((y_t
- mean(y))^2) of synthetic_returns.csv. This is deterministic weighted least squares, so
reproduction is machine-exact: interior ~1e-14, boundary ~1e-14 (p=1) to ~1e-12 (p=3, the cubic
design's floating-point summation order -- documented, not a convention gap). Clean-room (CLAUDE.md
12): built from Feng-Gries-Fritz 2020 / Beran-Feng 2002, validated against gsmooth OUTPUT, never
its Rcpp source.

Pinned conventions: m = floor(n*b) window half-width; kernel (1-u^2)^mu with scale s = max|offset|+1
(= m+1 interior); boundary "fixed" (bb=0) truncates the window, "knn" (bb=1) keeps 2m+1 points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import KERNELS, local_poly

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

_BB = {0: "fixed", 1: "knn"}  # smoots bb -> our boundary name


def _series() -> np.ndarray:  # type: ignore[type-arg]
    """The semiparametric scale input: log-squared demeaned returns of the synthetic series."""
    y = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    return np.log((y - y.mean()) ** 2)


def _meta() -> dict:  # type: ignore[type-arg]
    """The gsmooth fixture metadata (bandwidth, kernel map, ...)."""
    return json.loads((_FIXTURE_DIR / "smooth_gsmooth_meta.json").read_text(encoding="utf-8"))


def _fixture(p: int, mu: int, bb: int) -> np.ndarray:  # type: ignore[type-arg]
    """Load the gsmooth m_hat fixture for a (p, mu, bb) convention."""
    return np.loadtxt(_FIXTURE_DIR / f"smooth_gsmooth_p{p}_mu{mu}_bb{bb}.csv", skiprows=1)


# --------------------------------------------------------------------------- #
# Machine-exact reproduction of all 16 fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bb", [0, 1])
@pytest.mark.parametrize("mu", [0, 1, 2, 3])
@pytest.mark.parametrize("p", [1, 3])
def test_local_poly_matches_gsmooth_fixture(p: int, mu: int, bb: int) -> None:
    """local_poly reproduces smoots::gsmooth for every (p, mu, bb) convention (machine-exact)."""
    b = _meta()["bandwidth_b"]
    m_hat = local_poly(_series(), p=p, mu=mu, bandwidth=b, boundary=_BB[bb])
    fixture = _fixture(p, mu, bb)
    # p=3's cubic Vandermonde loosens the boundary to FP-summation-order ~1e-12; p=1 stays ~1e-13.
    tol = 5e-12 if p == 3 else 1e-12
    assert np.max(np.abs(m_hat - fixture)) < tol


def test_local_poly_interior_is_tightest() -> None:
    """Interior (shift-invariant) rows reproduce to ~1e-13; boundary carries the p=3 FP noise."""
    series = _series()
    n = series.size
    b = _meta()["bandwidth_b"]
    m = int(np.floor(n * b))
    for p in (1, 3):
        m_hat = local_poly(series, p=p, mu=1, bandwidth=b, boundary="knn")
        interior_dev = np.max(np.abs(m_hat - _fixture(p, 1, 1))[m : n - m])
        assert interior_dev < 1e-13


# --------------------------------------------------------------------------- #
# Convention sanity checks
# --------------------------------------------------------------------------- #


def test_boundary_only_affects_the_ends() -> None:
    """ "fixed" and "knn" give identical interior rows -- bb only changes the boundary treatment."""
    series = _series()
    n = series.size
    m = int(np.floor(n * _meta()["bandwidth_b"]))
    fixed = local_poly(series, p=1, mu=1, boundary="fixed")
    knn = local_poly(series, p=1, mu=1, boundary="knn")
    assert np.max(np.abs(fixed - knn)[m : n - m]) < 1e-13  # interior identical
    assert np.max(np.abs(fixed - knn)[:m]) > 1e-3  # boundary genuinely differs


def test_uniform_linear_interior_is_the_moving_average() -> None:
    """mu=0 (uniform) + p=1 (local-linear): the interior estimate is the symmetric-window mean."""
    series = _series()
    n = series.size
    m = int(np.floor(n * 0.15))
    m_hat = local_poly(series, p=1, mu=0, bandwidth=0.15, boundary="fixed")
    # Interior local-linear with uniform weights on a symmetric window -> the plain window mean.
    t = 1000
    window_mean = series[t - m : t + m + 1].mean()
    assert m_hat[t] == pytest.approx(window_mean, abs=1e-12)


def test_epanechnikov_kernel_integrates_to_one() -> None:
    """Sanity on the kernel family: the normalized (1-u^2)^mu constants integrate to 1."""
    from scipy import integrate

    norm_consts = {0: 0.5, 1: 0.75, 2: 15.0 / 16.0, 3: 35.0 / 32.0}
    for mu, c in norm_consts.items():
        val, _ = integrate.quad(lambda u, mu=mu, c=c: c * (1.0 - u * u) ** mu, -1.0, 1.0)
        assert val == pytest.approx(1.0, abs=1e-12)
    assert set(norm_consts) == set(KERNELS)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_local_poly_rejects_bad_args() -> None:
    """Invalid p (even/non-positive), mu, bandwidth and boundary are rejected."""
    series = _series()
    with pytest.raises(ValueError, match="odd"):
        local_poly(series, p=2)
    with pytest.raises(ValueError, match="mu"):
        local_poly(series, p=1, mu=5)
    with pytest.raises(ValueError, match="bandwidth"):
        local_poly(series, p=1, bandwidth=0.6)
    with pytest.raises(ValueError, match="boundary"):
        local_poly(series, p=1, boundary="extend")  # type: ignore[arg-type]
