r"""Option-book P\&L — the bridge between the pricing and risk pillars.

An :class:`OptionBook` holds positions (instrument + pricing engine + quantity,
plus an optional underlying leg for hedges) over one underlying, and turns a set of
:class:`MarketScenarios` into a **P\&L distribution** three ways:

* **Full revaluation** — reprice the whole book through the attached pricing
  engines under each scenario and difference against the base value. Exact up to
  the engines themselves (no drift between the risk path and the pricing path: it
  *is* the pricing path), but costs one full repricing per scenario.
* **Delta-normal** — first-order approximation
  :math:`\Delta\,\delta S \;(+\; \nu\,\delta\sigma)` from the book's aggregate
  Greeks. One Greek computation, then vectorised: fast, and the industry's classic
  variance--covariance treatment. Linear in the risk factors, so it is *blind to
  gamma* — for a short-gamma book it under-states tail risk, and for a long-gamma
  book it over-states it.
* **Delta-gamma** — adds the second-order spot term
  :math:`\tfrac12\,\Gamma\,\delta S^2`, repairing most of the curvature error at
  almost no extra cost.

All three are evaluated on the *same* scenario set, so any divergence is
approximation error, not sampling noise — that comparison ("when is the fast
approximation safe?") is the model-validation deliverable, demonstrated in
``tests/risk/test_derivatives.py`` and ``scripts/derivatives_var_report.py``.

Scenarios are **instantaneous** market shocks (spot returns, optional additive vol
shifts): time is not rolled forward, so theta drops out and the delta/gamma
comparison is exact to its own order. The P\&L vector feeds the existing risk layer
unchanged — ``empirical_var_es(-pnl, level)`` — which is what the
:mod:`~quantica.risk.portfolio` P\&L-series seam was built for.

Book Greeks are computed by **central-difference bump-and-reval through each
position's own engine**, so the approximations stay consistent with whatever
numerical method prices the book (the analytic Greeks equal these bumps for
European positions — validated in Phase 1). Deterministic engines (analytic, tree,
PDE) are recommended: a Monte-Carlo engine draws fresh randoms on every call, which
adds sampling noise to bumped Greeks and scenario P\&L (though seeded runs stay
reproducible end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from quantica.risk.measures import RiskEstimate, empirical_var_es

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.pricing.engines import PricingEngine
    from quantica.pricing.instruments import VanillaOption
    from quantica.pricing.processes import BlackScholesProcess

__all__ = [
    "BookGreeks",
    "BookPosition",
    "MarketScenarios",
    "OptionBook",
    "book_var_es",
]

RevaluationMethod = Literal["full", "delta-normal", "delta-gamma"]

# Named bump sizes for the finite-difference book Greeks (CLAUDE.md §6). Relative
# spot bumps keep the step scale-free; 1e-4 balances truncation against round-off
# for smooth engines. Lattice/PDE engines have discretisation kinks — pass a larger
# bump for those.
_DEFAULT_REL_SPOT_BUMP = 1e-4
_DEFAULT_VOL_BUMP = 1e-4


class BookGreeks(NamedTuple):
    r"""Aggregate book sensitivities used by the P\&L approximations.

    ``delta``/``gamma`` are w.r.t. the underlying spot; ``vega`` is per unit of
    volatility (matching :class:`~quantica.pricing.greeks.Greeks` conventions).
    """

    delta: float
    gamma: float
    vega: float


@dataclass(frozen=True)
class BookPosition:
    """One book line: an instrument, the engine that prices it, and a signed quantity.

    Parameters
    ----------
    instrument : VanillaOption
        Any option the attached engine can price (European/American/exotic).
    engine : PricingEngine
        The numerical method used to (re)price this position. Deterministic
        engines are recommended (see module docstring).
    quantity : float
        Signed position size (negative = short). Must be non-zero and finite.
    """

    instrument: VanillaOption
    engine: PricingEngine
    quantity: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.quantity) or self.quantity == 0.0:
            raise ValueError(f"quantity must be finite and non-zero, got {self.quantity}")


@dataclass(frozen=True)
class MarketScenarios:
    r"""A set of instantaneous market shocks: spot returns and optional vol shifts.

    Parameters
    ----------
    spot_returns : ndarray
        Relative spot moves :math:`r_i` (the shocked spot is :math:`S(1+r_i)`).
        Every ``1 + r_i`` must be positive.
    vol_shifts : ndarray, optional
        Additive shifts to the Black--Scholes volatility, aligned with
        ``spot_returns``. The shocked vol is floored at zero (a documented clamp:
        a negative total vol is meaningless, and ``vol == 0`` is a valid limit).
    """

    spot_returns: FloatArray
    vol_shifts: FloatArray | None = None

    def __post_init__(self) -> None:
        r = np.asarray(self.spot_returns, dtype=np.float64)
        if r.ndim != 1 or r.size == 0:
            raise ValueError("spot_returns must be a non-empty 1-D array")
        if np.any(1.0 + r <= 0.0):
            raise ValueError("every scenario must keep the spot positive (1 + r > 0)")
        object.__setattr__(self, "spot_returns", r)
        if self.vol_shifts is not None:
            v = np.asarray(self.vol_shifts, dtype=np.float64)
            if v.shape != r.shape:
                raise ValueError(
                    f"vol_shifts must match spot_returns shape {r.shape}, got {v.shape}"
                )
            object.__setattr__(self, "vol_shifts", v)

    @property
    def n_scenarios(self) -> int:
        """Number of scenarios."""
        return int(np.asarray(self.spot_returns).size)

    @classmethod
    def generate(
        cls,
        n_scenarios: int,
        rng: np.random.Generator,
        *,
        spot_vol: float,
        drift: float = 0.0,
        vol_shift_vol: float = 0.0,
    ) -> MarketScenarios:
        r"""Draw seeded Gaussian scenarios (the simulated-market case).

        Parameters
        ----------
        n_scenarios : int
            Number of draws.
        rng : numpy.random.Generator
            Seeded generator, injected for reproducibility.
        spot_vol : float
            Standard deviation of the spot return over the risk horizon (e.g. a
            daily vol for one-day VaR).
        drift : float, optional
            Mean spot return over the horizon (default 0).
        vol_shift_vol : float, optional
            Standard deviation of the additive vol shift; ``0`` (default) means
            spot-only scenarios.
        """
        if n_scenarios < 1:
            raise ValueError(f"n_scenarios must be at least 1, got {n_scenarios}")
        if spot_vol < 0.0 or vol_shift_vol < 0.0:
            raise ValueError("spot_vol and vol_shift_vol must be non-negative")
        returns = rng.normal(drift, spot_vol, n_scenarios)
        # Truncate pathological draws so the shocked spot stays positive (returns
        # below -100% have no meaning for a price).
        returns = np.maximum(returns, -0.999)
        shifts = rng.normal(0.0, vol_shift_vol, n_scenarios) if vol_shift_vol > 0.0 else None
        return cls(spot_returns=returns, vol_shifts=shifts)


@dataclass(frozen=True)
class OptionBook:
    r"""A book of option positions (plus an optional underlying leg) on one underlying.

    Parameters
    ----------
    positions : sequence of BookPosition
        The option lines. May be empty only if ``underlying_quantity`` is non-zero.
    process : BlackScholesProcess
        The *base* market state (spot, rate, dividend, vol) the book is marked on;
        scenarios shock this process.
    underlying_quantity : float, optional
        A signed quantity of the underlying itself (delta one, gamma/vega zero) —
        the natural way to express delta-hedged books.
    """

    positions: tuple[BookPosition, ...]
    process: BlackScholesProcess
    underlying_quantity: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
        if not self.positions and self.underlying_quantity == 0.0:
            raise ValueError("the book is empty: no positions and no underlying")
        if not np.isfinite(self.underlying_quantity):
            raise ValueError("underlying_quantity must be finite")

    # ------------------------------------------------------------------ value

    def value(self, process: BlackScholesProcess | None = None) -> float:
        """Mark the whole book (options plus underlying leg) under ``process``.

        Defaults to the base process. Repricing goes through each position's own
        engine, so this is by construction the same number the pricing pillar
        produces — the no-drift guarantee.
        """
        proc = process if process is not None else self.process
        option_value = sum(
            pos.quantity * pos.engine.calculate(pos.instrument, proc) for pos in self.positions
        )
        return float(option_value + self.underlying_quantity * proc.spot)

    def _shocked_process(self, spot_return: float, vol_shift: float) -> BlackScholesProcess:
        """The base process under one scenario (vol floored at zero, documented)."""
        shocked = self.process.with_spot(self.process.spot * (1.0 + spot_return))
        if vol_shift != 0.0:
            shocked = shocked.with_vol(max(self.process.vol + vol_shift, 0.0))
        return shocked

    # ------------------------------------------------------------------ greeks

    def greeks(
        self,
        *,
        rel_spot_bump: float = _DEFAULT_REL_SPOT_BUMP,
        vol_bump: float = _DEFAULT_VOL_BUMP,
    ) -> BookGreeks:
        r"""Aggregate book delta/gamma/vega by central-difference bump-and-reval.

        The bumps run through each position's *own* engine, so the Greeks are
        consistent with the same numerics that full revaluation uses. For smooth
        engines the defaults are near-optimal; lattice/PDE engines may need a
        larger ``rel_spot_bump`` to step over discretisation kinks.
        """
        if rel_spot_bump <= 0.0 or vol_bump <= 0.0:
            raise ValueError("bump sizes must be positive")
        S = self.process.spot
        h = rel_spot_bump * S
        base = self.value()
        up = self.value(self.process.with_spot(S + h))
        down = self.value(self.process.with_spot(S - h))
        delta = (up - down) / (2.0 * h)
        gamma = (up - 2.0 * base + down) / (h * h)

        vol = self.process.vol
        vol_lo = max(vol - vol_bump, 0.0)
        v_up = self.value(self.process.with_vol(vol + vol_bump))
        v_down = self.value(self.process.with_vol(vol_lo))
        vega = (v_up - v_down) / (vol + vol_bump - vol_lo)
        return BookGreeks(delta=float(delta), gamma=float(gamma), vega=float(vega))

    # ------------------------------------------------------------------ P&L

    def full_revaluation_pnl(self, scenarios: MarketScenarios) -> FloatArray:
        r"""Exact scenario P\&L: reprice the book under every scenario and difference.

        One full book repricing per scenario — the cost of exactness. Any
        nonlinearity the engines can see (gamma, vol convexity, barriers, early
        exercise) is captured.
        """
        base = self.value()
        shifts = self._vol_shifts(scenarios)
        pnl = np.empty(scenarios.n_scenarios, dtype=np.float64)
        for i, (r, dv) in enumerate(zip(scenarios.spot_returns, shifts, strict=True)):
            pnl[i] = self.value(self._shocked_process(float(r), float(dv))) - base
        return pnl

    def delta_normal_pnl(
        self, scenarios: MarketScenarios, *, greeks: BookGreeks | None = None
    ) -> FloatArray:
        r"""First-order P\&L: :math:`\Delta\,\delta S + \nu\,\delta\sigma`.

        The vega term appears only when the scenarios carry vol shifts (the
        standard first-order-in-all-risk-factors treatment). Linear ⇒ blind to
        gamma; see the module docstring for the direction of the resulting bias.
        """
        g = greeks if greeks is not None else self.greeks()
        d_spot = self.process.spot * scenarios.spot_returns
        pnl = g.delta * d_spot + g.vega * self._vol_shifts(scenarios)
        return np.asarray(pnl, dtype=np.float64)

    def delta_gamma_pnl(
        self, scenarios: MarketScenarios, *, greeks: BookGreeks | None = None
    ) -> FloatArray:
        r"""Second-order P\&L: adds :math:`\tfrac12\,\Gamma\,\delta S^2` to delta-normal."""
        g = greeks if greeks is not None else self.greeks()
        d_spot = self.process.spot * scenarios.spot_returns
        pnl = self.delta_normal_pnl(scenarios, greeks=g) + 0.5 * g.gamma * d_spot * d_spot
        return np.asarray(pnl, dtype=np.float64)

    def _vol_shifts(self, scenarios: MarketScenarios) -> FloatArray:
        """The scenarios' vol shifts, or zeros for spot-only scenario sets."""
        if scenarios.vol_shifts is None:
            return np.zeros(scenarios.n_scenarios, dtype=np.float64)
        return np.asarray(scenarios.vol_shifts, dtype=np.float64)


def book_var_es(
    book: OptionBook,
    scenarios: MarketScenarios,
    level: float,
    *,
    method: RevaluationMethod = "full",
) -> RiskEstimate:
    r"""VaR/ES of an option book under scenarios, by the chosen revaluation method.

    A thin adapter into the existing risk layer: compute the book's scenario P\&L,
    negate it into losses, and hand it to
    :func:`~quantica.risk.measures.empirical_var_es` — the risk/backtest machinery
    is untouched (the P\&L-series seam doing its job). Evaluating all three methods
    on the *same* scenario set isolates approximation error from sampling noise.
    """
    if method == "full":
        pnl = book.full_revaluation_pnl(scenarios)
    elif method == "delta-normal":
        pnl = book.delta_normal_pnl(scenarios)
    elif method == "delta-gamma":
        pnl = book.delta_gamma_pnl(scenarios)
    else:
        raise ValueError(f"method must be 'full', 'delta-normal' or 'delta-gamma', got {method!r}")
    label = "full-revaluation" if method == "full" else method
    return empirical_var_es(-pnl, level, method=label)
