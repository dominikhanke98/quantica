r"""Merton (1976) jump-diffusion European pricing, two independent ways.

Merton adds a compound-Poisson jump component to the Black--Scholes diffusion (see
:class:`~quantica.pricing.processes.MertonProcess`). European options admit **two**
independent valuations, which cross-check each other with no external reference:

Closed form — a Poisson-weighted sum of Black--Scholes prices
-------------------------------------------------------------
Conditional on exactly :math:`n` jumps occurring before expiry, the terminal
log-spot is again normal, so the option is an ordinary Black--Scholes option with
an inflated variance and a shifted forward. Summing over the Poisson law of the
jump count :math:`N_T \sim \mathrm{Poisson}(\lambda T)` gives

.. math::

    V = \sum_{n=0}^{\infty} \frac{e^{-\lambda T}(\lambda T)^n}{n!}\;
        \mathrm{BS}\!\left(S_0, K, r, q_n, \sigma_n, T\right),

with :math:`\sigma_n^2 = \sigma^2 + n\sigma_J^2/T` and an effective dividend
:math:`q_n = q + \lambda\bar\kappa - n(\mu_J + \sigma_J^2/2)/T` chosen so that each
term's forward matches the true conditional forward while the discount stays the
genuine :math:`e^{-rT}`. The Black--Scholes term is delegated to
:class:`~quantica.pricing.engines.analytic.AnalyticEuropeanEngine`, so this pricer
inherits that engine's validation. The series is truncated once the remaining
Poisson tail cannot move the price by more than a documented tolerance.

Characteristic function + FFT
-----------------------------
The Merton CF of :math:`\ln S_T` is the product of the diffusion CF and the
compound-Poisson CF; it plugs straight into the shared Carr--Madan transform
(:func:`~quantica.pricing.engines._carr_madan.carr_madan_call_price`) reused from
the Heston engine.

References
----------
Merton, R. (1976). "Option pricing when underlying stock returns are
discontinuous", *J. Financial Economics*.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import ExerciseStyle, OptionType
from quantica.pricing.engines._carr_madan import carr_madan_call_price
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.instruments import EuropeanOption, VanillaOption
from quantica.pricing.processes import BlackScholesProcess, MertonProcess

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_ALPHA = 1.5  # Carr--Madan damping (calls need alpha > 0)
_DEFAULT_N_FFT = 4096  # FFT length (power of two)
_DEFAULT_ETA = 0.25  # integration-grid spacing in the Fourier variable
_DEFAULT_SERIES_TOL = 1e-12  # target truncation error on the Poisson sum (price units)
_MAX_SERIES_TERMS = 2000  # hard cap on the number of Poisson terms

# One shared, stateless analytic engine for the per-jump Black--Scholes terms.
_ANALYTIC = AnalyticEuropeanEngine()


def merton_characteristic_function(
    u: NDArray[np.complex128],
    tau: float,
    *,
    rate: float,
    div: float,
    vol: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    spot: float,
) -> NDArray[np.complex128]:
    r"""Characteristic function of :math:`\ln S_\tau` under Merton jump-diffusion.

    Returns :math:`\varphi(u) = \mathbb E[e^{i u \ln S_\tau}]`. As
    :math:`\lambda \to 0` this reduces to the Gaussian (Black--Scholes) CF.
    """
    iu = 1j * u
    kbar = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift = math.log(spot) + (rate - div - lam * kbar - 0.5 * vol * vol) * tau
    diffusion = -0.5 * vol * vol * (u * u) * tau
    jump = lam * tau * (np.exp(iu * mu_j - 0.5 * sigma_j * sigma_j * (u * u)) - 1.0)
    result: NDArray[np.complex128] = np.exp(iu * drift + diffusion + jump)
    return result


def merton_jump_price(
    instrument: VanillaOption,
    process: MertonProcess,
    *,
    tol: float = _DEFAULT_SERIES_TOL,
    max_terms: int = _MAX_SERIES_TERMS,
) -> float:
    r"""Closed-form Merton price: the Poisson-weighted sum of Black--Scholes prices.

    Parameters
    ----------
    instrument : VanillaOption
        A European option (call or put).
    process : MertonProcess
        The jump-diffusion parameters and market.
    tol : float, optional
        Target truncation error in price units. Terms are added until the
        remaining Poisson tail mass, bounded by ``max(S, K)``, is below ``tol``.
        Pass ``tol = 0`` to force exactly ``max_terms + 1`` terms (used to
        demonstrate series convergence).
    max_terms : int, optional
        Hard cap on the number of jump terms (``n = 0 .. max_terms``).
    """
    S = process.spot
    K = instrument.strike
    r = process.rate
    q = process.div
    T = instrument.expiry
    sigma = process.vol
    lam = process.lam
    sigma_j = process.sigma_j

    if T <= 0.0:
        return instrument.payoff(S)  # type: ignore[return-value]

    kbar = process.compensator
    ln_1_plus_kbar = process.mu_j + 0.5 * sigma_j * sigma_j  # = ln(1 + kbar)
    lam_t = lam * T
    option = EuropeanOption(strike=K, expiry=T, option_type=instrument.option_type)

    # Truncate when the remaining Poisson tail cannot move the price by > tol.
    # A BS call term is bounded by S and a put by K e^{-rT} <= K, so the tail-price
    # error is at most max(S, K) times the tail probability.
    tail_threshold = tol / max(S, K, 1.0)

    total = 0.0
    weight = math.exp(-lam_t)  # Poisson(lam_t) mass at n = 0
    cumulative = 0.0
    for n in range(max_terms + 1):
        var_n = sigma * sigma + n * sigma_j * sigma_j / T
        sigma_n = math.sqrt(var_n)
        q_n = q + lam * kbar - n * ln_1_plus_kbar / T
        bs = _ANALYTIC.calculate(option, BlackScholesProcess(spot=S, rate=r, div=q_n, vol=sigma_n))
        total += weight * bs
        cumulative += weight
        if (1.0 - cumulative) < tail_threshold:
            break
        weight *= lam_t / (n + 1)  # recurrence for e^{-lam_t}(lam_t)^{n+1}/(n+1)!
    return total


class MertonClosedFormEngine:
    """Merton pricer via the Poisson-weighted Black--Scholes series (closed form).

    Parameters
    ----------
    tol : float, optional
        Target series-truncation error in price units (default ``1e-12``).
    max_terms : int, optional
        Hard cap on the number of Poisson terms (default ``2000``).

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol,
    pricing a European option under a :class:`MertonProcess`.
    """

    def __init__(
        self, *, tol: float = _DEFAULT_SERIES_TOL, max_terms: int = _MAX_SERIES_TERMS
    ) -> None:
        if tol < 0.0:
            raise ValueError(f"tol must be non-negative, got {tol}")
        if max_terms < 0:
            raise ValueError(f"max_terms must be non-negative, got {max_terms}")
        self.tol = tol
        self.max_terms = max_terms

    def calculate(self, instrument: VanillaOption, process: MertonProcess) -> float:
        """Present value of ``instrument`` under the Merton ``process``."""
        _check(instrument, process)
        return merton_jump_price(instrument, process, tol=self.tol, max_terms=self.max_terms)


class MertonFFTEngine:
    """Merton pricer via the Carr--Madan FFT of the characteristic function.

    Parameters
    ----------
    alpha : float, optional
        Carr--Madan damping factor (default 1.5; must be positive).
    n_fft : int, optional
        FFT length (default 4096; a power of two).
    eta : float, optional
        Spacing of the Fourier-integration grid (default 0.25).

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol,
    pricing a European option under a :class:`MertonProcess`. Puts come from
    put--call parity (model-independent).
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

    def calculate(self, instrument: VanillaOption, process: MertonProcess) -> float:
        """Present value of ``instrument`` under the Merton ``process``."""
        _check(instrument, process)
        K = instrument.strike
        T = instrument.expiry

        def cf(u: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return merton_characteristic_function(
                u,
                T,
                rate=process.rate,
                div=process.div,
                vol=process.vol,
                lam=process.lam,
                mu_j=process.mu_j,
                sigma_j=process.sigma_j,
                spot=process.spot,
            )

        call = carr_madan_call_price(
            cf, K, T, process.rate, alpha=self.alpha, n_fft=self.n_fft, eta=self.eta
        )
        if instrument.option_type is OptionType.CALL:
            return call
        # Put via parity (model-independent): P = C - S e^{-qT} + K e^{-rT}.
        forward_leg = process.spot * math.exp(-process.div * T)
        strike_leg = K * math.exp(-process.rate * T)
        return float(call - forward_leg + strike_leg)


def _check(instrument: VanillaOption, process: MertonProcess) -> None:
    """Shared guardrails for the Merton engines."""
    if not isinstance(process, MertonProcess):
        raise TypeError(f"Merton engines require a MertonProcess, got {type(process).__name__}")
    if instrument.exercise is not ExerciseStyle.EUROPEAN:
        raise ValueError("the Merton engines price European exercise only")
