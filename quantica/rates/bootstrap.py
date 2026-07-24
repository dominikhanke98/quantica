r"""Sequential bootstrap — build a discount curve so every input reprices to par.

The bootstrap solves the pillars shortest-first: each instrument adds one discount factor,
chosen so it prices to par given the curve already built. A deposit fixes its pillar in closed
form (:meth:`~quantica.rates.instruments.Deposit.par_discount_factor`); a swap's par condition
also depends, *through the interpolation*, on the discount factors at its intermediate coupon
dates, so its pillar is found by a 1-D root solve (``scipy.optimize`` — the one place the
bootstrap needs it, CLAUDE.md §3).

**Iterative refinement (why one pass is not enough for global interpolation).** With a *local*
scheme (linear, log-linear) each pillar depends only on its neighbours, so a single sequential
pass reprices every input exactly. With a *non-local* scheme (a cubic spline), adding a later
pillar changes the interpolant at an earlier swap's intermediate coupon dates and nudges it off
par. So after the sequential pass we **sweep** the swap pillars, re-solving each against the
full curve, until every instrument reprices to par to machine precision — the standard
iterative bootstrap. The finished curve then reprices **every** input to ~1e-13 *whatever* the
interpolation, while the schemes still disagree about the forwards *between* the pillars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from quantica.rates.curve import DiscountCurve
from quantica.rates.instruments import Deposit

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.rates.curve import CurveInterpolation
    from quantica.rates.instruments import RateInstrument

__all__ = ["bootstrap"]

_MIN_DF = 1.0e-8  # lower bracket for the discount-factor root solve
_PAR_TOL = 1.0e-13  # convergence tolerance on the repricing residual
_MAX_SWEEPS = 50


def bootstrap(
    instruments: list[RateInstrument],
    interpolation: CurveInterpolation | None = None,
) -> DiscountCurve:
    """Bootstrap a :class:`~quantica.rates.curve.DiscountCurve` from market instruments.

    Parameters
    ----------
    instruments : list of RateInstrument
        Deposits and/or swaps; sorted internally by maturity and solved shortest-first. The
        maturities must be strictly increasing after sorting (one pillar per instrument).
    interpolation : CurveInterpolation, optional
        The interpolation scheme for the curve being built (default log-linear on discount
        factors).

    Returns
    -------
    DiscountCurve
        The curve that reprices every input instrument to par (to ~1e-13, refined
        iteratively for non-local interpolation schemes).

    Raises
    ------
    ValueError
        If ``instruments`` is empty or two instruments share a maturity.
    """
    if not instruments:
        raise ValueError("need at least one instrument to bootstrap")
    ordered = sorted(instruments, key=lambda inst: inst.maturity)
    times = np.array([inst.maturity for inst in ordered], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("instruments must have strictly increasing (distinct) maturities")

    # Sequential first pass: each pillar solved against the curve built so far.
    dfs = np.ones(len(ordered), dtype=np.float64)
    for k, inst in enumerate(ordered):
        if isinstance(inst, Deposit):
            dfs[k] = inst.par_discount_factor()
        else:
            dfs[: k + 1] = _solve_pillar(inst, times[: k + 1], dfs[: k + 1], k, interpolation)

    # Refinement sweeps: re-solve each swap pillar against the *full* curve until every
    # instrument reprices to par (needed only for non-local interpolation; a no-op otherwise).
    for _ in range(_MAX_SWEEPS):
        for k, inst in enumerate(ordered):
            if not isinstance(inst, Deposit):
                dfs = _solve_pillar(inst, times, dfs, k, interpolation)
        curve = DiscountCurve(times, dfs, interpolation)
        if max(abs(inst.value(curve)) for inst in ordered) < _PAR_TOL:
            break

    return DiscountCurve(times, dfs, interpolation)


def _solve_pillar(
    inst: RateInstrument,
    times: FloatArray,
    dfs: FloatArray,
    k: int,
    interpolation: CurveInterpolation | None,
) -> FloatArray:
    """Root-solve pillar ``k``'s discount factor so ``inst`` prices to par; returns new dfs."""
    trial = dfs.copy()

    def residual(df_new: float) -> float:
        trial[k] = df_new
        return inst.value(DiscountCurve(times, trial, interpolation))

    upper = 1.0 - 1e-12
    trial[k] = float(brentq(residual, _MIN_DF, upper, xtol=1e-15, rtol=1e-15))
    return trial
