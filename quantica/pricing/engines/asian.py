r"""Monte Carlo engine for Asian (average-price) options, with a geometric control.

An Asian option pays on the *average* of the underlying over a set of monitoring
dates :math:`t_1 < \dots < t_n = T` rather than the terminal spot. Two averages:

* **Geometric** — :math:`G = (\prod_i S_{t_i})^{1/n}` is lognormal (a sum of
  normals in log-space), so it has a **closed form** (a Black--Scholes formula in
  an effective forward and volatility): see :func:`geometric_asian_price`. This
  is the analytic anchor for the whole family.
* **Arithmetic** — :math:`A = \tfrac1n \sum_i S_{t_i}` is a sum of lognormals,
  which is not lognormal and has **no closed form**, so it is priced by Monte
  Carlo.

Control variate (the highlight)
-------------------------------
The arithmetic and geometric averages of the *same* path are almost perfectly
correlated, and the geometric payoff has a known expectation (its closed form).
That makes the discounted geometric payoff an excellent control variate for the
arithmetic price:

.. math::

    \hat A_{\text{cv}} = \hat A - \beta\,(\hat G_{\text{MC}} - G_{\text{exact}}),

with :math:`\beta = \operatorname{Cov}(A, G)/\operatorname{Var}(G)` estimated from
the sample. The variance reduction is large (a factor of tens to hundreds),
and — unlike a generic control — it is motivated by a real financial
relationship between the two contracts.

Randomness is an injected, seeded :class:`numpy.random.Generator`; standard
errors are reported (numerical-validation skill §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from quantica.core.types import AveragingType, FloatArray, OptionType
from quantica.pricing.engines._common import unpack
from quantica.pricing.engines._paths import GBMPathSimulator
from quantica.pricing.engines.montecarlo import MCResult
from quantica.pricing.instruments import AsianOption

if TYPE_CHECKING:
    from quantica.pricing.processes import BlackScholesProcess

# Below this control variance the control carries no information; skip it.
_MIN_CONTROL_VAR = 1e-300


def geometric_asian_price(
    spot: float,
    strike: float,
    rate: float,
    div: float,
    vol: float,
    expiry: float,
    n_averaging_dates: int,
    option_type: OptionType,
) -> float:
    r"""Closed-form price of a discretely-monitored geometric-average Asian option.

    With monitoring times :math:`t_i = i\,T/n` (``i = 1..n``), :math:`\ln G` is
    normal with

    .. math::

        \mu_G = \ln S + (r - q - \tfrac12\sigma^2)\,\bar t, \qquad
        \sigma_G^2 = \frac{\sigma^2}{n^2} \sum_{i,j} \min(t_i, t_j),

    where :math:`\bar t` is the mean monitoring time, giving the Black--Scholes-like
    value :math:`e^{-rT}\,\omega\,[\,\mathbb E[G]\,N(\omega d_1) - K\,N(\omega d_2)\,]`.
    """
    n = n_averaging_dates
    omega = option_type.sign
    disc = float(np.exp(-rate * expiry))
    times = np.arange(1, n + 1) * (expiry / n)
    mu_g = float(np.log(spot) + (rate - div - 0.5 * vol * vol) * times.mean())
    var_g = float((vol * vol / (n * n)) * np.minimum.outer(times, times).sum())

    expected_g = float(np.exp(mu_g + 0.5 * var_g))
    if var_g <= _MIN_CONTROL_VAR:  # deterministic (zero vol / zero maturity)
        return disc * max(omega * (expected_g - strike), 0.0)

    sigma_g = np.sqrt(var_g)
    d1 = (mu_g + var_g - np.log(strike)) / sigma_g
    d2 = d1 - sigma_g
    return (
        disc
        * omega
        * (expected_g * float(norm.cdf(omega * d1)) - strike * float(norm.cdf(omega * d2)))
    )


class AsianMonteCarloEngine:
    """Monte Carlo pricer for an :class:`AsianOption` (arithmetic or geometric).

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    antithetic : bool, optional
        Mirror the Brownian increments (variance reduction).
    control_variate : bool, optional
        For an *arithmetic* Asian, use the geometric Asian (with its exact price)
        as a control variate. Ignored for a geometric Asian (which is priced
        directly and has its own closed form anyway).

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol
    (price only). Use :meth:`estimate` for the price *and* its standard error.
    """

    def __init__(
        self,
        n_paths: int,
        *,
        rng: np.random.Generator,
        antithetic: bool = False,
        control_variate: bool = False,
    ) -> None:
        self.n_paths = n_paths
        self.control_variate = control_variate
        self._sim = GBMPathSimulator(n_paths, rng=rng, antithetic=antithetic)

    def calculate(self, instrument: AsianOption, process: BlackScholesProcess) -> float:
        """Present value (the point estimate; see :meth:`estimate` for the SE)."""
        return self.estimate(instrument, process).price

    def estimate(self, instrument: AsianOption, process: BlackScholesProcess) -> MCResult:
        """Price ``instrument`` under ``process`` with its Monte Carlo standard error."""
        if not isinstance(instrument, AsianOption):
            raise TypeError(
                f"AsianMonteCarloEngine prices AsianOption, got {type(instrument).__name__}"
            )
        S, K, r, q, sigma, T, omega = unpack(instrument, process)
        n = instrument.n_averaging_dates
        dt = T / n
        disc = float(np.exp(-r * T))

        paths = self._sim.simulate(spot=S, rate=r, div=q, vol=sigma, dt=dt, n_steps=n)
        monitored = paths[:, 1:]  # S at t_1 .. t_n
        geometric_avg = np.exp(np.log(monitored).mean(axis=1))

        if instrument.averaging is AveragingType.GEOMETRIC:
            per_path = disc * np.maximum(omega * (geometric_avg - K), 0.0)
        else:
            arithmetic_avg = monitored.mean(axis=1)
            per_path = disc * np.maximum(omega * (arithmetic_avg - K), 0.0)
            if self.control_variate:
                geo_payoff = disc * np.maximum(omega * (geometric_avg - K), 0.0)
                control_mean = geometric_asian_price(
                    S, K, r, q, sigma, T, n, instrument.option_type
                )
                per_path = _apply_control_variate(per_path, geo_payoff, control_mean)

        samples = self._sim.combine_antithetic(per_path)
        price = float(samples.mean())
        std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
        return MCResult(price=price, std_error=std_error, n_paths=self.n_paths)


def _apply_control_variate(
    payoff: FloatArray, control: FloatArray, control_mean: float
) -> FloatArray:
    """Return ``payoff - beta (control - E[control])`` with the sample-optimal beta."""
    cov = np.cov(payoff, control, ddof=1)
    var_control = float(cov[1, 1])
    if var_control <= _MIN_CONTROL_VAR:
        return payoff
    beta = float(cov[0, 1]) / var_control
    adjusted: FloatArray = payoff - beta * (control - control_mean)
    return adjusted
