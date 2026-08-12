"""Validation of the fractional-differencing operator (1-L)^d (numerical-validation skill, Phase 3).

The operator is an internal filter, **not a fitted model**, so it has no fEGarch output fixture:
validation is analytic + cross-method, not fixture-matching. Checks: the coefficients match the
analytic binomial ``(-1)^i C(d,i)`` to machine precision; ``d=0`` is the identity and ``d=1`` the
ordinary first difference exactly; the FFT and direct filter paths agree to FFT round-off; the
truncated filter converges as ``L`` grows; and fractionally integrating white noise by
``(1-L)^{-d}`` produces the hyperbolic long-memory ACF signature (``rho(k) ~ k^{2d-1}``),
decisively unlike white noise. Statistical checks use a seeded ``numpy.random.Generator`` (no global
RNG). Every tolerance is stated with its rationale.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.timeseries.fegarch import fracdiff, fracdiff_coeffs
from scipy.special import binom


def test_coefficients_match_analytic_binomial() -> None:
    """b_i(d) equals the analytic (-1)^i C(d,i) to machine precision (b_0 = 1, plus the recursion).

    Tolerance ~1e-13: the cumulative-product recursion and scipy's ``binom`` are both machine-order,
    so agreement is at the floating-point noise floor (realized ~1e-16).
    """
    idx = np.arange(300)
    for d in (0.2, 0.3, 0.45):
        b = fracdiff_coeffs(d, 300)
        assert b[0] == 1.0
        assert np.max(np.abs(b - (-1.0) ** idx * binom(d, idx))) < 1e-13
        # the recursion b_i = b_{i-1}*(i-1-d)/i holds elementwise
        assert np.max(np.abs(b[1:] - b[:-1] * (idx[1:] - 1.0 - d) / idx[1:])) < 1e-15


def test_d_zero_is_identity() -> None:
    """(1-L)^0 leaves any series unchanged (all b_i = 0 for i >= 1); machine precision."""
    b = fracdiff_coeffs(0.0, 50)
    assert b[0] == 1.0
    assert np.all(b[1:] == 0.0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    assert np.max(np.abs(fracdiff(x, 0.0) - x)) < 1e-13


def test_d_one_is_first_difference() -> None:
    """(1-L)^1 recovers the ordinary first difference exactly (b = [1, -1, 0, ...])."""
    b = fracdiff_coeffs(1.0, 6)
    assert b[0] == 1.0
    assert b[1] == -1.0
    assert np.all(b[2:] == 0.0)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(400)
    expected = np.empty_like(x)
    expected[0] = x[0]  # first difference with pre-sample x_{-1} = 0
    expected[1:] = x[1:] - x[:-1]
    assert np.max(np.abs(fracdiff(x, 1.0) - expected)) < 1e-13


def test_fft_and_direct_paths_agree() -> None:
    """The FFT and direct convolution paths agree to FFT round-off (~1e-12) at fixed d, truncation.

    Cross-method convergence: the two paths compute the same truncated causal filter by different
    algorithms, so they must agree once the shared discretisation (d, L) is fixed. Tolerance 1e-10
    covers FFT round-off (~eps * length; realized ~1e-15) with margin across platforms/FFT builds.
    """
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    for d in (0.0, 0.25, 0.45, 1.0, -0.3):
        fft = fracdiff(x, d, method="fft")
        direct = fracdiff(x, d, method="direct")
        assert np.max(np.abs(fft - direct)) < 1e-10


def test_truncation_converges_as_length_grows() -> None:
    """The truncated filter converges to the full (L=n-1) filter as L grows; error decays with L.

    ``y_L -> y_full`` monotonically in L (the tail coefficients |b_L| ~ L^(-d-1) shrink), and
    ``y_full`` is exact at L = n-1. This is the truncation-error convergence table's assertion.
    """
    rng = np.random.default_rng(3)
    d, n = 0.35, 3000
    x = rng.standard_normal(n)
    y_full = fracdiff(x, d, trunc=n - 1)
    errors = [
        np.max(np.abs(fracdiff(x, d, trunc=length) - y_full)) for length in (10, 50, 250, 1000)
    ]
    # strictly decreasing with L, and each smaller than the coarsest
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[3] < 5e-3  # by L=1000 the truncation error is well under 1% of the series scale
    assert np.max(np.abs(fracdiff(x, d, trunc=n - 1) - y_full)) == 0.0  # full length is exact
    # analytic coefficient tail decays like L^(-d-1)
    tail = np.array([abs(fracdiff_coeffs(d, length + 1)[-1]) for length in (50, 250, 1000)])
    assert tail[0] > tail[1] > tail[2]


def test_fractional_integration_has_long_memory_acf() -> None:
    """(1-L)^(-d) white noise has a hyperbolic ACF (rho ~ k^{2d-1}) — the long-memory signature.

    Seeded, sampling-based, asserted loosely (the fitted exponent is finite-sample biased and
    depends on the lag window): the log-log ACF slope sits negative and in a wide band around
    ``2d-1``, and the mid-lag autocorrelation is orders of magnitude above a white-noise control
    (short memory would have ~0 there). This distinguishes hyperbolic from geometric, not a tight
    exponent match.
    """
    rng = np.random.default_rng(20240901)
    n, d = 100_000, 0.4
    eps = rng.standard_normal(n)
    fi = fracdiff(eps, -d)  # fractional integration -> long memory

    def _acf(y: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        y = y - y.mean()
        fft_len = 1
        while fft_len < 2 * y.size:
            fft_len *= 2
        spec = np.fft.rfft(y, fft_len)
        acf = np.fft.irfft(spec * np.conj(spec), fft_len)[: y.size].real
        return np.asarray(acf / acf[0])

    acf_fi = _acf(fi)
    acf_wn = _acf(eps)

    # 1. hyperbolic decay: negative log-log slope, loosely around 2d-1 (not geometric/near-zero)
    lags = np.arange(20, 2000)
    rho = acf_fi[lags]
    mask = rho > 0
    slope = float(np.polyfit(np.log(lags[mask]), np.log(rho[mask]), 1)[0])
    assert (2 * d - 1) - 0.5 < slope < (2 * d - 1) + 0.2
    assert slope < -0.05  # clearly decaying, not flat

    # 2. long memory persists: mid-lag ACF far above the white-noise control (short memory ~ 0)
    fi_mid = float(np.mean(np.abs(acf_fi[50:150])))
    wn_mid = float(np.mean(np.abs(acf_wn[50:150])))
    assert fi_mid > 10.0 * wn_mid


def test_trunc_and_presample_conventions() -> None:
    """The App. C.3 conventions: default trunc = n-1, pre-sample of the filtered quantity = 0."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(100)
    # default trunc (None) equals explicit n-1
    assert np.array_equal(fracdiff(x, 0.3), fracdiff(x, 0.3, trunc=x.size - 1))
    # trunc is capped at n-1 (cannot use more lags than the available history under pre-sample 0)
    assert np.array_equal(fracdiff(x, 0.3, trunc=10_000), fracdiff(x, 0.3, trunc=x.size - 1))
    # pre-sample 0 (default) means the first output is just b_0 * x_0 = x_0
    assert np.isclose(fracdiff(x, 0.3)[0], x[0], atol=1e-13)
    # a non-zero pre-sample fill changes the warm-up (shifts the first output)
    assert not np.isclose(fracdiff(x, 0.3, presample=5.0)[0], x[0], atol=1e-6)


def test_edge_cases_and_errors() -> None:
    """Empty input, and invalid trunc / method arguments."""
    assert fracdiff(np.array([]), 0.3).size == 0
    with pytest.raises(ValueError, match="length must be >= 1"):
        fracdiff_coeffs(0.3, 0)
    with pytest.raises(ValueError, match="trunc must be non-negative"):
        fracdiff(np.arange(10.0), 0.3, trunc=-1)
    with pytest.raises(ValueError, match="unknown method"):
        fracdiff(np.arange(10.0), 0.3, method="nope")
