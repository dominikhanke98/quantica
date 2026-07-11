r"""Shared Carr--Madan (1999) FFT transform for characteristic-function pricing.

Both the Heston and Merton European engines price by the same device: given the
characteristic function :math:`\varphi(u) = \mathbb E[e^{iu\ln S_T}]` of log-spot,
the damped call price is a Fourier integral evaluated by one FFT. The transform
itself is model-agnostic — it takes a CF *callable* — so it is factored out here
once a second engine (Merton) needed it (CLAUDE.md §2, "extract on the second
consumer").

With damping factor :math:`\alpha > 0` the damped call transform is

.. math::

    \psi(v) = \frac{e^{-rT}\,\varphi\big(v - (\alpha+1)i\big)}
                   {\alpha^2 + \alpha - v^2 + i(2\alpha+1)v},

and :math:`C(k) = \tfrac{e^{-\alpha k}}{\pi}\int_0^\infty \mathrm{Re}
\big(e^{-ivk}\psi(v)\big)\,dv`. We place the requested log-strike *exactly on an
FFT node* (so there is no interpolation error) and read the price there.

References
----------
Carr, P. & Madan, D. (1999). "Option valuation using the fast Fourier
transform", *J. Computational Finance*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import FloatArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    #: A characteristic function of log-spot: maps a complex argument array to
    #: :math:`\varphi(u) = \mathbb E[e^{iu\ln S_T}]`.
    CharacteristicFunction = Callable[[NDArray[np.complex128]], NDArray[np.complex128]]


def carr_madan_call_price(
    cf: CharacteristicFunction,
    strike: float,
    expiry: float,
    rate: float,
    *,
    alpha: float,
    n_fft: int,
    eta: float,
) -> float:
    """Carr--Madan FFT price of a European call for a given characteristic function.

    Parameters
    ----------
    cf : callable
        The characteristic function of :math:`\\ln S_T` (already carrying the
        model parameters and maturity), evaluated on a complex argument array.
    strike, expiry, rate : float
        Option strike ``K``, maturity ``T``, and the risk-free rate ``r`` used for
        the :math:`e^{-rT}` discount factor.
    alpha : float
        Damping factor (``> 0``); the price is theoretically independent of it.
    n_fft : int
        FFT length (a power of two).
    eta : float
        Spacing of the Fourier-integration grid.
    """
    n = n_fft
    lam = 2.0 * np.pi / (n * eta)  # log-strike grid spacing
    # Centre the log-strike grid so ln(K) is exactly node n//2 (no interpolation).
    k_target = np.log(strike)
    k_min = k_target - (n // 2) * lam

    j = np.arange(n)
    v = eta * j
    u_arg = np.asarray(v - (alpha + 1.0) * 1j, dtype=np.complex128)
    phi = cf(u_arg)
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-rate * expiry) * phi / denom

    simpson = _simpson_weights(n) * (eta / 3.0)
    integrand = np.exp(-1j * v * k_min) * psi * simpson
    transform = np.fft.fft(integrand)

    k_grid = k_min + lam * j
    call_curve = np.exp(-alpha * k_grid) / np.pi * transform.real
    return float(call_curve[n // 2])


def _simpson_weights(n: int) -> FloatArray:
    """Composite Simpson coefficients (1, 4, 2, 4, ..., 4, 1)."""
    weights = np.ones(n)
    weights[1::2] = 4.0
    weights[2:-1:2] = 2.0
    return weights
