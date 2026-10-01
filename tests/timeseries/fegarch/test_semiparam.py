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
from quantica.timeseries.fegarch import (
    KERNELS,
    bartlett_variance_factor,
    integrated_squared_derivative,
    local_poly,
)

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
    with pytest.raises(ValueError, match="p >= v"):
        local_poly(series, p=2)
    with pytest.raises(ValueError, match="mu"):
        local_poly(series, p=1, mu=5)
    with pytest.raises(ValueError, match="bandwidth"):
        local_poly(series, p=1, bandwidth=0.6)
    with pytest.raises(ValueError, match="boundary"):
        local_poly(series, p=1, boundary="extend")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Phase-7 sub-build 2 machine-exact components: derivative local-poly + SM c_f
# --------------------------------------------------------------------------- #


def _deriv_meta() -> dict:  # type: ignore[type-arg]
    """The derivative-smoother fixture metadata (bandwidth, mu)."""
    return json.loads((_FIXTURE_DIR / "smooth_deriv_meta.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("v", "p", "bb", "tol"),
    [(2, 3, 1, 1e-7), (2, 3, 0, 1e-7), (4, 5, 1, 1e-2)],
)
def test_local_poly_derivative_matches_gsmooth(v: int, p: int, bb: int, tol: float) -> None:
    """local_poly(v=k) = beta[v]*v!*n^v reproduces smoots::gsmooth(v=k); n^v amplifies FP error."""
    b = _deriv_meta()["bandwidth_b"]
    mk = local_poly(_series(), v=v, p=p, mu=1, bandwidth=b, boundary=_BB[bb])
    fixture = np.loadtxt(_FIXTURE_DIR / f"smooth_deriv_v{v}_p{p}_bb{bb}.csv", skiprows=1)
    # n^v (= n^2, n^4) amplifies the ~1e-15 relative WLS error; relative agreement stays ~1e-8
    # even for the ill-conditioned 4th-derivative (p=5) solve.
    rel = np.max(np.abs(mk - fixture)) / np.max(np.abs(fixture))
    assert rel < 1e-8
    assert np.max(np.abs(mk - fixture)) < tol


def test_sm_variance_factor_genuine_reconstruction() -> None:
    """SM c_f: recompute residuals from the series at b0, then Bartlett(given M) == cf0 exactly."""
    meta = json.loads((_FIXTURE_DIR / "smooth_cf_sm_meta.json").read_text(encoding="utf-8"))
    b0 = meta["b0"]
    m_window = meta["Mcf_NP"]["L0_opt"]
    cf0 = meta["Mcf_NP"]["cf0"]
    series = _series()
    # Genuine clean-room recon: our own trend -> our own residuals -> the Bartlett sum.
    residuals = series - local_poly(series, p=1, mu=1, bandwidth=b0, boundary="knn")
    res_fixture = np.loadtxt(_FIXTURE_DIR / "smooth_cf_sm_res.csv", skiprows=1)
    assert np.max(np.abs(residuals - res_fixture)) < 1e-12  # our residuals match smoots'
    assert bartlett_variance_factor(residuals, window=m_window) == pytest.approx(cf0, abs=1e-12)


def test_sm_variance_factor_weight_convention() -> None:
    """The (M+1) weight is the implementation convention; the paper's (M+0.5) would NOT match."""
    meta = json.loads((_FIXTURE_DIR / "smooth_cf_sm_meta.json").read_text(encoding="utf-8"))
    b0, m_window, cf0 = meta["b0"], meta["Mcf_NP"]["L0_opt"], meta["Mcf_NP"]["cf0"]
    res = _series() - local_poly(_series(), p=1, mu=1, bandwidth=b0, boundary="knn")
    n = res.size
    r = res - res.mean()
    gamma = np.array([np.sum(r[: n - j] * r[j:]) / n for j in range(m_window + 1)])
    paper = gamma[0] + 2 * sum(
        (1 - j / (m_window + 0.5)) * gamma[j] for j in range(1, m_window + 1)
    )
    assert abs(paper - cf0) > 1e-3  # the paper's (M+0.5) weights are the wrong convention here
    assert bartlett_variance_factor(res, window=m_window) == pytest.approx(cf0, abs=1e-12)


def test_integrated_squared_derivative_mechanism() -> None:
    """I[m^(k)] = trapezoidal int of {m^(k)}^2 over [0.05,0.95] -- positive, finite, monotone."""
    series = _series()
    full = integrated_squared_derivative(series, k=2, p=3, mu=1, bandwidth=0.25)
    assert np.isfinite(full) and full > 0.0
    # Narrower interior limits integrate a strict sub-interval -> strictly smaller.
    inner = integrated_squared_derivative(series, k=2, p=3, mu=1, bandwidth=0.25, cb=0.2, db=0.8)
    assert 0.0 < inner < full
    with pytest.raises(ValueError, match="cb"):
        integrated_squared_derivative(series, k=2, p=3, cb=0.5, db=0.5)


def test_bartlett_rejects_bad_window() -> None:
    """The Bartlett window must satisfy 0 <= M < n."""
    res = _series()
    with pytest.raises(ValueError, match="window"):
        bartlett_variance_factor(res, window=-1)
    with pytest.raises(ValueError, match="window"):
        bartlett_variance_factor(res, window=res.size)
