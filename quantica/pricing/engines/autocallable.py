r"""Autocallable notes: Monte Carlo across Black--Scholes, Heston and Merton.

The autocallable (:class:`~quantica.pricing.instruments.AutocallableNote`) is a
*composition* of machinery already validated elsewhere in the package rather than new
numerics:

* the **full-path simulators** in :mod:`quantica.pricing.engines._paths`
  (:class:`~quantica.pricing.engines._paths.GBMPathSimulator` for Black--Scholes,
  :class:`~quantica.pricing.engines._paths.MertonPathSimulator`,
  :class:`~quantica.pricing.engines._paths.HestonPathSimulator`), each anchored to its
  transform/closed-form pricer by a European cross-check;
* the **discrete-monitoring** pattern from the barrier engine — observe the underlying on
  a fixed schedule and act only at those dates.

The payoff wiring is the new part. On each observation date :math:`t_i` (1-indexed), if
:math:`S_{t_i} \ge b_{\text{auto}} S_0` the note redeems early for
:math:`N\,(1 + i\,c)` and the path terminates. A path that never triggers is settled at
maturity against the European downside barrier: :math:`N` if
:math:`S_T \ge b_{\text{down}} S_0`, else :math:`N\,S_T/S_0` (the short down-and-in put).

Discrete monitoring — genuinely discrete, no continuous-limit bias
------------------------------------------------------------------
A continuously-monitored barrier (the :mod:`~quantica.pricing.engines.barrier` step)
needs a Brownian-bridge correction because Monte Carlo misses between-step crossings. An
autocallable does **not**: its observation dates are *contractual*, so the discrete grid
*is* the product. Under Black--Scholes and Merton the marginals are exact at every
observation, so there is **no** discretisation bias to remove — only Monte Carlo error.
Under Heston the sole bias is the Euler variance-discretisation of the simulator
(controlled by ``heston_substeps``), not the monitoring.

The smile matters (the headline)
--------------------------------
The holder is short a down-and-in put — short volatility and short skew. A flat-vol
Black--Scholes price therefore *misprices* the note versus a smile-consistent Heston
price; :func:`~quantica.pricing.engines.autocallable` users can quantify the gap with the
reproduction script. Randomness is an injected, seeded
:class:`numpy.random.Generator`; the standard error is reported (numerical-validation
skill §5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantica.core.types import FloatArray
from quantica.pricing.engines._paths import (
    GBMPathSimulator,
    HestonPathSimulator,
    MertonPathSimulator,
)
from quantica.pricing.instruments import AutocallableNote
from quantica.pricing.processes import (
    BlackScholesProcess,
    HestonProcess,
    MertonProcess,
)

_Process = BlackScholesProcess | HestonProcess | MertonProcess


@dataclass(frozen=True)
class AutocallableResult:
    """Monte Carlo valuation of an autocallable, with its redemption diagnostics.

    Parameters
    ----------
    price : float
        Present value of the note (same units as the note's ``notional``).
    std_error : float
        Monte Carlo standard error of ``price``.
    n_paths : int
        Number of simulated paths.
    autocall_probabilities : ndarray, shape (n_observations,)
        Probability the note first autocalls on each observation date.
    maturity_probability : float
        Probability the note survives to maturity without autocalling.
    loss_probability : float
        Probability of a capital loss (survives to maturity *and* finishes below the
        downside barrier). ``autocall_probabilities.sum() + maturity_probability`` is
        ``1`` by construction.
    """

    price: float
    std_error: float
    n_paths: int
    autocall_probabilities: FloatArray
    maturity_probability: float
    loss_probability: float


class AutocallableMonteCarloEngine:
    """Monte Carlo pricer for an :class:`AutocallableNote` under BS, Heston or Merton.

    The process type selects the path simulator; the payoff logic is identical across all
    three. See the module docstring for the discrete-monitoring reasoning.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    heston_substeps : int, optional
        Euler sub-steps per observation for the Heston simulator (default 32); ignored for
        the exact Black--Scholes and Merton simulators.

    Notes
    -----
    Satisfies the price-only engine contract via :meth:`calculate`; use :meth:`estimate`
    for the price, its standard error, and the autocall-probability breakdown.
    """

    def __init__(
        self,
        n_paths: int,
        *,
        rng: np.random.Generator,
        heston_substeps: int = 32,
    ) -> None:
        self.n_paths = n_paths
        self.rng = rng
        self.heston_substeps = heston_substeps

    def calculate(self, instrument: AutocallableNote, process: _Process) -> float:
        """Present value of ``instrument`` (the point estimate; see :meth:`estimate`)."""
        return self.estimate(instrument, process).price

    def estimate(self, instrument: AutocallableNote, process: _Process) -> AutocallableResult:
        """Price ``instrument`` under ``process`` with SE and autocall diagnostics."""
        if not isinstance(instrument, AutocallableNote):
            raise TypeError(
                "AutocallableMonteCarloEngine prices AutocallableNote, got "
                f"{type(instrument).__name__}"
            )
        spot = process.spot
        rate = process.rate
        n = instrument.n_observations
        dt = instrument.maturity / n
        observed = self._simulate(process, dt=dt, n_steps=n)  # (n_paths, n + 1)

        obs = observed[:, 1:]  # levels at t_1 .. t_n
        times = instrument.observation_times
        disc = np.exp(-rate * times)  # (n,)

        auto_level = instrument.autocall_barrier * spot
        hit = obs >= auto_level  # (n_paths, n)
        autocalled = hit.any(axis=1)
        first_hit = hit.argmax(axis=1)  # first True per path (0 if none — masked below)

        notional = instrument.notional
        payoff = np.empty(self.n_paths, dtype=np.float64)

        # Early-redemption legs: notional * (1 + i*coupon), discounted to t_i (i 1-indexed).
        idx = first_hit[autocalled]
        payoff[autocalled] = notional * (1.0 + (idx + 1.0) * instrument.coupon) * disc[idx]

        # Survivors settle at maturity against the European downside barrier.
        survivors = ~autocalled
        final = obs[survivors, -1]
        prot_level = instrument.downside_barrier * spot
        redemption = np.where(final >= prot_level, notional, notional * final / spot)
        payoff[survivors] = redemption * disc[-1]

        price = float(payoff.mean())
        std_error = float(payoff.std(ddof=1) / np.sqrt(self.n_paths))

        # Diagnostics: first-autocall probability per date, survival, and loss.
        autocall_probs = np.array(
            [float(np.mean(autocalled & (first_hit == i))) for i in range(n)],
            dtype=np.float64,
        )
        maturity_prob = float(np.mean(survivors))
        loss_prob = float(np.mean(survivors & (obs[:, -1] < prot_level)))

        return AutocallableResult(
            price=price,
            std_error=std_error,
            n_paths=self.n_paths,
            autocall_probabilities=autocall_probs,
            maturity_probability=maturity_prob,
            loss_probability=loss_prob,
        )

    def _simulate(self, process: _Process, *, dt: float, n_steps: int) -> FloatArray:
        """Dispatch to the path simulator matching ``process`` and return observed paths."""
        if isinstance(process, BlackScholesProcess):
            sim = GBMPathSimulator(self.n_paths, rng=self.rng)
            return sim.simulate(
                spot=process.spot,
                rate=process.rate,
                div=process.div,
                vol=process.vol,
                dt=dt,
                n_steps=n_steps,
            )
        if isinstance(process, MertonProcess):
            merton = MertonPathSimulator(self.n_paths, rng=self.rng)
            return merton.simulate(
                spot=process.spot,
                rate=process.rate,
                div=process.div,
                vol=process.vol,
                lam=process.lam,
                mu_j=process.mu_j,
                sigma_j=process.sigma_j,
                dt=dt,
                n_steps=n_steps,
            )
        heston = HestonPathSimulator(self.n_paths, rng=self.rng, n_substeps=self.heston_substeps)
        return heston.simulate(
            spot=process.spot,
            rate=process.rate,
            div=process.div,
            v0=process.v0,
            kappa=process.kappa,
            theta=process.theta,
            xi=process.xi,
            rho=process.rho,
            dt=dt,
            n_steps=n_steps,
        )
