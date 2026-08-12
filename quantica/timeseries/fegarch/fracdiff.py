r"""Fractional-differencing operator :math:`(1-L)^d` — the long-memory engine (Phase 3).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (the binomial expansion of :math:`(1-L)^d`, Bollerslev-Mikkelsen 1996 / Hosking 1981; WP 2026-04
    App. C.3 for the truncation / pre-sample policy, which cites Nielsen-Noël 2021 for the FFT),
    never the `fEGarch` source. This is an **internal filter, not a fitted model**, so it has no
    output fixture; correctness is established analytically and by cross-method agreement (the
    numerical-validation skill), not by fixture-matching.

The fractional-differencing operator is the truncated binomial (infinite-order MA) expansion

.. math::

    (1 - L)^d = \sum_{i=0}^{\infty} b_i\,L^i,\qquad
    b_0 = 1,\quad b_i = b_{i-1}\,\frac{i - 1 - d}{i} = (-1)^i \binom{d}{i},

for any real :math:`d` (:math:`d > 0` fractional differencing, :math:`d < 0` fractional
integration). Applied to a series :math:`(x_t)`, the filtered value is the causal convolution
:math:`y_t = \sum_{i=0}^{L} b_i\,x_{t-i}` with the **pre-sample convention** :math:`x_t = 0` for
:math:`t \le 0`.

**fEGarch truncation / pre-sample policy (WP 2026-04 App. C.3) — Phase 4 inherits this.** The
default truncation length is :math:`L = n - 1` (so the infinite series is truncated exactly as far
back as needed for the first observation to be included), and pre-sample values of the filtered
quantity are :math:`0` for :math:`t \le 0`. Every fractionally-integrated Phase-4 model
(FIEGARCH / FIMEGARCH / FIMLog-GARCH / FIGARCH / …) composes its recursion with this operator under
**exactly this convention**, so it is load-bearing here even without a fixture: a wrong truncation
policy would propagate to all of them.

Two filter paths are provided: a **direct** convolution (``O(nL)``) and an **FFT** convolution
(``O(n \log n)``, the performance path the long-memory models will use, per the Nielsen-Noël 2021
FFT approach App. C.3 cites). They agree to FFT round-off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import fft as sp_fft

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "fracdiff",
    "fracdiff_coeffs",
]


def fracdiff_coeffs(d: float, length: int) -> FloatArray:
    r"""Binomial coefficients :math:`b_i(d)` of :math:`(1-L)^d`, for ``i = 0 .. length-1``.

    Uses the exact recursion :math:`b_0 = 1`, :math:`b_i = b_{i-1}(i-1-d)/i` (equivalently
    :math:`b_i = (-1)^i \binom{d}{i}`), evaluated as a cumulative product.

    Parameters
    ----------
    d : float
        The fractional order (any real; ``> 0`` differencing, ``< 0`` integration).
    length : int
        Number of coefficients to return (``>= 1``); ``b_0 .. b_{length-1}``.

    Returns
    -------
    ndarray, shape (length,)
        The coefficients, with ``b[0] == 1``.

    Raises
    ------
    ValueError
        If ``length < 1``.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    b = np.empty(length, dtype=np.float64)
    b[0] = 1.0
    if length > 1:
        i = np.arange(1, length, dtype=np.float64)
        b[1:] = np.cumprod((i - 1.0 - d) / i)
    return b


def _convolve_direct(x: FloatArray, b: FloatArray) -> FloatArray:
    """Direct-convolution causal filter (``O(nL)``): ``y_t = sum_i b_i x_{t-i}``, pre-sample 0."""
    return np.asarray(np.convolve(x, b)[: x.size], dtype=np.float64)


def _convolve_fft(x: FloatArray, b: FloatArray) -> FloatArray:
    """FFT-convolution causal filter (``O(n log n)``): the linear-convolution head of ``x * b``."""
    m = x.size + b.size - 1
    fft_len = sp_fft.next_fast_len(m)
    spectrum = np.fft.rfft(x, fft_len) * np.fft.rfft(b, fft_len)
    return np.asarray(np.fft.irfft(spectrum, fft_len)[: x.size], dtype=np.float64)


def fracdiff(
    x: FloatArray,
    d: float,
    *,
    trunc: int | None = None,
    presample: float = 0.0,
    method: str = "fft",
) -> FloatArray:
    r"""Apply :math:`(1-L)^d` to ``x`` under the fEGarch truncation / pre-sample convention.

    Computes the causal, truncated filter :math:`y_t = \sum_{i=0}^{\min(L,\,t-1)} b_i\,x_{t-i}` with
    pre-sample values :math:`x_t = ` ``presample`` for :math:`t \le 0` (WP 2026-04 App. C.3: the
    pre-sample of the filtered quantity is ``0``).

    Parameters
    ----------
    x : ndarray, shape (n,)
        The series to filter.
    d : float
        The fractional order (``> 0`` differencing, ``< 0`` integration).
    trunc : int, optional
        Truncation length :math:`L` of the coefficient series. Default (``None``) is the App. C.3
        long-memory default :math:`L = n - 1`; values are capped at ``n - 1`` (no more lags than the
        available history under the pre-sample-0 convention).
    presample : float, optional
        The constant pre-sample value of ``x`` for :math:`t \le 0` (default ``0.0``, the App. C.3
        convention). A non-zero value shifts the warm-up of the filter.
    method : str, optional
        ``"fft"`` (default, ``O(n log n)``) or ``"direct"`` (``O(nL)``); agree to FFT round-off.

    Returns
    -------
    ndarray, shape (n,)
        The fractionally-differenced series.

    Raises
    ------
    ValueError
        If ``trunc`` is negative or ``method`` is unknown.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()
    if trunc is not None and trunc < 0:
        raise ValueError("trunc must be non-negative")
    length = (n - 1) if trunc is None else min(int(trunc), n - 1)
    b = fracdiff_coeffs(d, length + 1)

    # Pre-sample values of x for t <= 0 are `presample`; subtract so the convolution sees zeros,
    # then add back the deterministic filter of that constant. For presample == 0 (the App. C.3
    # default) this reduces to a plain causal convolution. The add-back is the constant
    # `presample * sum(b_0..b_L)`: at every t the missing pre-sample terms plus the subtracted
    # in-sample constant recombine to the same partial coefficient sum (see the module note).
    shifted = x - presample
    if method == "fft":
        y = _convolve_fft(shifted, b)
    elif method == "direct":
        y = _convolve_direct(shifted, b)
    else:
        raise ValueError(f"unknown method {method!r} (use 'fft' or 'direct')")
    if presample != 0.0:
        y = y + presample * float(np.sum(b))
    return y
