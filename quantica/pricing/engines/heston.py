r"""Heston European pricing via the characteristic function and the Carr--Madan FFT.

The Heston log-spot has a known characteristic function
:math:`\varphi(u) = \mathbb E[e^{i u \ln S_T}]`, and Carr & Madan (1999) turn the
option price into a Fourier integral of that CF, evaluated by FFT.

Branch-cut stability (the "little Heston trap")
-----------------------------------------------
The CF involves a complex square root :math:`d` and a ratio :math:`g`. The
*naive* choice :math:`g = (\beta + d)/(\beta - d)` with a :math:`+d\tau`
exponent crosses the branch cut of the complex logarithm for long maturities,
producing discontinuities that wreck the integration. The **little Heston trap**
(Albrecher et al. 2007) uses :math:`g = (\beta - d)/(\beta + d)` with a
:math:`-d\tau` exponent, which stays inside the principal branch and is stable
for all maturities. We use it from the start (see :func:`heston_characteristic_function`).

Carr--Madan
-----------
With damping factor :math:`\alpha > 0`, the damped call transform is

.. math::

    \psi(v) = \frac{e^{-rT}\,\varphi\big(v - (\alpha+1)i\big)}
                   {\alpha^2 + \alpha - v^2 + i(2\alpha+1)v},

and :math:`C(k) = \tfrac{e^{-\alpha k}}{\pi}\int_0^\infty \mathrm{Re}
\big(e^{-ivk}\psi(v)\big)\,dv`. We place the requested log-strike *exactly on an
FFT node* (so there is no interpolation error) and read the price there. Puts
come from put--call parity, which holds model-independently.

References
----------
Carr, P. & Madan, D. (1999). "Option valuation using the fast Fourier
transform", *J. Computational Finance*. Albrecher et al. (2007), "The little
Heston trap", *Wilmott*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import ExerciseStyle, OptionType
from quantica.pricing.engines._carr_madan import carr_madan_call_price
from quantica.pricing.instruments import VanillaOption
from quantica.pricing.processes import HestonProcess

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_ALPHA = 1.5  # Carr--Madan damping (calls need alpha > 0)
_DEFAULT_N_FFT = 4096  # FFT length (power of two)
_DEFAULT_ETA = 0.25  # integration-grid spacing in the Fourier variable
_XI_FLOOR = 1e-8  # below this vol-of-vol, use the deterministic-variance CF


def heston_characteristic_function(
    u: NDArray[np.complex128],
    tau: float,
    *,
    rate: float,
    div: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    spot: float,
) -> NDArray[np.complex128]:
    r"""Characteristic function of :math:`\ln S_\tau` under Heston (little-trap form).

    Returns :math:`\varphi(u) = \mathbb E[e^{i u \ln S_\tau}]`. In the
    :math:`\xi \to 0` limit the variance is deterministic and this reduces to the
    Gaussian (Black--Scholes) CF with integrated variance
    :math:`\int_0^\tau v_t\,dt`.
    """
    iu = 1j * u
    if xi < _XI_FLOOR:
        # Deterministic variance v_t = theta + (v0 - theta) e^{-kappa t}.
        integrated_var = (
            theta * tau + (v0 - theta) * (1.0 - np.exp(-kappa * tau)) / kappa
            if kappa > 0.0
            else v0 * tau
        )
        drift = np.log(spot) + (rate - div) * tau
        limit_cf: NDArray[np.complex128] = np.exp(iu * drift - 0.5 * (u * u + iu) * integrated_var)
        return limit_cf

    beta = kappa - rho * xi * iu
    d = np.sqrt(beta * beta + xi * xi * (u * u + iu))
    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * tau)
    xi2 = xi * xi

    big_d = (beta - d) / xi2 * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    big_c = (rate - div) * iu * tau + (kappa * theta / xi2) * (
        (beta - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    result: NDArray[np.complex128] = np.exp(big_c + big_d * v0 + iu * np.log(spot))
    return result


class HestonFFTEngine:
    """Heston European pricer via the Carr--Madan FFT.

    Parameters
    ----------
    alpha : float, optional
        Carr--Madan damping factor (default 1.5). Must be positive; the price is
        theoretically independent of it, so its choice is a numerical knob whose
        stability is a validation target, not a magic number.
    n_fft : int, optional
        FFT length (default 4096; a power of two).
    eta : float, optional
        Spacing of the Fourier-integration grid (default 0.25).

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol
    (price only), pricing a European option under a :class:`HestonProcess`.
    """

    def __init__(
        self,
        *,
        alpha: float = _DEFAULT_ALPHA,
        n_fft: int = _DEFAULT_N_FFT,
        eta: float = _DEFAULT_ETA,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if n_fft < 2:
            raise ValueError(f"n_fft must be at least 2, got {n_fft}")
        if eta <= 0.0:
            raise ValueError(f"eta must be positive, got {eta}")
        self.alpha = alpha
        self.n_fft = n_fft
        self.eta = eta

    def calculate(self, instrument: VanillaOption, process: HestonProcess) -> float:
        """Present value of ``instrument`` under the Heston ``process``."""
        if not isinstance(process, HestonProcess):
            raise TypeError(
                f"HestonFFTEngine requires a HestonProcess, got {type(process).__name__}"
            )
        if instrument.exercise is not ExerciseStyle.EUROPEAN:
            raise ValueError("the Heston FFT engine prices European exercise only")

        K = instrument.strike
        T = instrument.expiry
        call = self._carr_madan_call(K, T, process)
        if instrument.option_type is OptionType.CALL:
            return call
        # Put via parity (model-independent): P = C - S e^{-qT} + K e^{-rT}.
        forward_leg = process.spot * np.exp(-process.div * T)
        strike_leg = K * np.exp(-process.rate * T)
        return float(call - forward_leg + strike_leg)

    def _carr_madan_call(self, strike: float, expiry: float, process: HestonProcess) -> float:
        def cf(u: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return heston_characteristic_function(
                u,
                expiry,
                rate=process.rate,
                div=process.div,
                v0=process.v0,
                kappa=process.kappa,
                theta=process.theta,
                xi=process.xi,
                rho=process.rho,
                spot=process.spot,
            )

        return carr_madan_call_price(
            cf, strike, expiry, process.rate, alpha=self.alpha, n_fft=self.n_fft, eta=self.eta
        )
