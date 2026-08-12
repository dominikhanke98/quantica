#!/usr/bin/env python
"""Generate the fractional-differencing validation table for the README.

The operator ``(1-L)^d`` (:mod:`quantica.timeseries.fegarch.fracdiff`) is an internal filter, not a
fitted model, so it has **no fEGarch output fixture**. This script is its validation artifact in
place of a fixture-match, per the numerical-validation skill: it reports (1) coefficient accuracy
against the analytic binomial ``(-1)^i C(d,i)``; (2) FFT-vs-direct cross-method agreement; (3) the
truncation-error convergence as the length ``L`` grows, alongside the analytic coefficient tail
``|b_L| ~ L^(-d-1)``; and (4) the long-memory ACF decay exponent of a fractionally-integrated
series, which should approach the ``2d-1`` hyperbolic signature (not geometric).

All statistical rows use an explicit, seeded ``numpy.random.Generator`` (never the global RNG), so
the table reproduces byte-for-byte. Regenerate with::

    python scripts/fracdiff_convergence.py

Do not hand-edit the numbers in the README -- rerun this script instead.
"""

from __future__ import annotations

import numpy as np
from quantica.timeseries.fegarch import fracdiff, fracdiff_coeffs
from scipy.special import binom

_SEED = 20240901


def _coefficient_accuracy() -> list[str]:
    """Rows: max |b_i(d) - (-1)^i C(d,i)| over the first 200 coefficients, for several d."""
    rows = ["### Coefficient accuracy (vs analytic (-1)^i C(d,i))", ""]
    rows.append("| d | b_0 | max abs error | recursion b_i/b_{i-1} - (i-1-d)/i |")
    rows.append("|---|---|---|---|")
    idx = np.arange(200)
    for d in (0.2, 0.3, 0.45):
        b = fracdiff_coeffs(d, 200)
        analytic = (-1.0) ** idx * binom(d, idx)
        rec = np.max(np.abs(b[1:] / b[:-1] - (idx[1:] - 1.0 - d) / idx[1:]))
        rows.append(f"| {d} | {b[0]:.1f} | {np.max(np.abs(b - analytic)):.2e} | {rec:.2e} |")
    return rows


def _fft_vs_direct(rng: np.random.Generator) -> list[str]:
    """Rows: max |FFT path - direct path| on a random series, for several d."""
    rows = ["", "### FFT-vs-direct cross-method agreement (random series, n=4000)", ""]
    rows.append("| d | max abs difference |")
    rows.append("|---|---|")
    x = rng.standard_normal(4000)
    for d in (0.0, 0.25, 0.45, 1.0):
        diff = np.max(np.abs(fracdiff(x, d, method="fft") - fracdiff(x, d, method="direct")))
        rows.append(f"| {d} | {diff:.2e} |")
    return rows


def _truncation_error(rng: np.random.Generator) -> list[str]:
    """Rows: ||y_L - y_full||_inf and coefficient tail |b_L| vs L (d=0.35, n=4000)."""
    rows = ["", "### Truncation-error convergence (d=0.35, n=4000)", ""]
    rows.append("| L | max|y_L - y_full| | |b_L| | L^(-d-1) ref |")
    rows.append("|---|---|---|---|")
    d, n = 0.35, 4000
    x = rng.standard_normal(n)
    y_full = fracdiff(x, d, trunc=n - 1)
    for length in (10, 50, 250, 1000, n - 1):
        err = np.max(np.abs(fracdiff(x, d, trunc=length) - y_full))
        b_tail = abs(fracdiff_coeffs(d, length + 1)[-1])
        rows.append(f"| {length} | {err:.2e} | {b_tail:.2e} | {length ** (-d - 1.0):.2e} |")
    return rows


def _acf_decay(rng: np.random.Generator) -> list[str]:
    """Rows: fitted log-log ACF slope of an FI(d) series vs the 2d-1 long-memory signature."""
    rows = ["", "### Long-memory ACF decay of (1-L)^(-d) white noise (n=200000)", ""]
    rows.append("| d | fitted log-log slope | 2d-1 (theory) | rho(1000) [long-memory persists] |")
    rows.append("|---|---|---|---|")
    n = 200_000
    for d in (0.3, 0.4):
        eps = rng.standard_normal(n)
        y = fracdiff(eps, -d)
        y = y - y.mean()
        fft_len = 1
        while fft_len < 2 * n:
            fft_len *= 2
        spec = np.fft.rfft(y, fft_len)
        acf = np.fft.irfft(spec * np.conj(spec), fft_len)[:n].real
        acf /= acf[0]
        lags = np.arange(20, 2000)
        rho = acf[lags]
        mask = rho > 0
        slope = float(np.polyfit(np.log(lags[mask]), np.log(rho[mask]), 1)[0])
        rows.append(f"| {d} | {slope:+.3f} | {2 * d - 1:+.3f} | {acf[1000]:.3f} |")
    return rows


def main() -> None:
    """Print the full fractional-differencing validation table as GitHub-flavoured Markdown."""
    rng = np.random.default_rng(_SEED)
    lines = ["# Fractional-differencing `(1-L)^d` -- validation table", ""]
    lines += _coefficient_accuracy()
    lines += _fft_vs_direct(rng)
    lines += _truncation_error(rng)
    lines += _acf_decay(rng)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
