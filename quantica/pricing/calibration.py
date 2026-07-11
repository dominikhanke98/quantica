r"""Calibration of the Heston model to a vanilla implied-volatility surface.

Given a set of market vanilla quotes — a grid of strikes :math:`\times` maturities,
each with an observed Black--Scholes implied volatility — recover the five Heston
parameters :math:`(v_0, \kappa, \theta, \xi, \rho)` that best reprice the surface.
The fit is a **nonlinear least-squares** problem solved with
:func:`scipy.optimize.least_squares` (trust-region reflective); the demonstrable
skill here is the *design* of the calibration — objective, weighting, parameter
bounds, the Feller diagnostic, identifiability profiling, and multi-start
robustness — not reimplementing the optimizer (CLAUDE.md §3).

Vol space vs price space (the default choice)
---------------------------------------------
Residuals can be measured in **implied-volatility points** (default) or in
**price**. Vol space is the economically sensible default: a fixed price error is
a *huge* vol error for a cheap out-of-the-money option and a *tiny* one for an
expensive in-the-money option, so a price-space fit silently overweights deep ITM
quotes (which carry the least information about the smile). Weighting the fit in
vol points treats each quote on the scale a trader actually cares about. We move
between price and vol space by reusing the step-3 implied-volatility solver
(:func:`~quantica.pricing.volatility.implied_volatility`), inverting each model
price on the *out-of-the-money* option at that strike, where vega is largest and
the inversion is best-conditioned.

Identifiability (named, not hidden)
-----------------------------------
A single surface does not pin all five parameters equally. :math:`\kappa`
(mean-reversion speed) and :math:`\theta` (long-run variance) trade off — many
:math:`(\kappa, \theta)` pairs produce almost the same term structure of variance
— so the objective is *flat* along that direction and :math:`\kappa` in particular
is only loosely determined. :math:`v_0` (short-dated level), :math:`\rho` (skew)
and :math:`\xi` (smile curvature) are usually pinned much more tightly. We surface
this two ways rather than pretend the fit is unique:

* :func:`profile_objective` pins one parameter across a range, re-optimises the
  other four, and returns the minimised RMSE at each pin — a direct read-out of
  the objective's *flatness* along each axis (flat ⇒ unidentified).
* ``calibrate_heston(..., n_starts>1)`` runs from several starting points and
  reports the parameter spread across the near-optimal fits
  (:attr:`HestonCalibrationResult.param_spread`) — a robustness check that the
  same basin is found from different starts.

Feller condition (reported, optionally penalised)
-------------------------------------------------
The Feller condition :math:`2\kappa\theta \ge \xi^2` guarantees the variance stays
strictly positive. A calibrated fit that violates it is still a valid Heston
model (the variance can touch zero), so by default we **report** the condition via
:attr:`HestonProcess.feller_satisfied` rather than force it. A soft penalty is
available (``feller_weight > 0``) for callers who want to bias the fit toward the
Feller-satisfying region; the trade-off (a cleaner variance process vs a slightly
worse surface fit) is theirs to make.

References
----------
Heston, S. (1993). "A closed-form solution for options with stochastic
volatility". Gatheral, J. (2006). *The Volatility Surface*. Cui, Y. et al. (2017),
"Full and fast calibration of the Heston stochastic volatility model".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from scipy.optimize import least_squares

from quantica.core.types import FloatArray, OptionType
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.engines.heston import HestonFFTEngine
from quantica.pricing.instruments import EuropeanOption
from quantica.pricing.processes import BlackScholesProcess, HestonProcess, Market
from quantica.pricing.volatility import implied_volatility

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator

__all__ = [
    "DEFAULT_BOUNDS",
    "HestonCalibrationResult",
    "HestonParams",
    "ObjectiveProfile",
    "ParamBounds",
    "VolQuote",
    "calibrate_heston",
    "profile_objective",
    "vol_surface_from_grid",
]

CalibrationSpace = Literal["vol", "price"]
ParamName = Literal["v0", "kappa", "theta", "xi", "rho"]

# Numerical parameters (named, not magic — CLAUDE.md §6).
_DEFAULT_MULTISTART_SEED = 0  # seed for the multi-start sampler when no rng given
_SPREAD_REL_TOL = 0.05  # a start counts as "near-optimal" if cost <= best*(1+this)
_IV_PRICE_FLOOR = 1e-12  # floor a model price before IV inversion (band guard)
_BAND_SHRINK = 1.0 - 1e-9  # keep a clamped price just inside the upper no-arb bound
# Penalty residual for a parameter set the pricer cannot evaluate (a non-finite
# price from an extreme, unphysical corner of the box). Huge next to the ~1e-2
# scale of a vol residual, so the optimizer is pushed firmly out of that corner.
_BAD_RESIDUAL = 1e3

# Shared stateless analytic engine for market-price / IV conversions.
_ANALYTIC = AnalyticEuropeanEngine()


class HestonParams(NamedTuple):
    r"""The five Heston parameters as a plain, ordered bundle.

    Order matches the optimizer's parameter vector:
    :math:`(v_0, \kappa, \theta, \xi, \rho)`.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    @classmethod
    def from_array(cls, x: Sequence[float] | FloatArray) -> HestonParams:
        """Build from an array-like, coercing entries to plain ``float``."""
        return cls(*(float(v) for v in x))

    def to_process(self, market: Market) -> HestonProcess:
        """Attach the parameters to a :class:`Market` to make a :class:`HestonProcess`."""
        return HestonProcess.from_market(
            market, v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=self.rho
        )

    @property
    def feller_satisfied(self) -> bool:
        r"""Whether the Feller condition :math:`2\kappa\theta \ge \xi^2` holds."""
        return 2.0 * self.kappa * self.theta >= self.xi * self.xi


#: Index of each parameter in the optimizer vector (and in :class:`HestonParams`).
_PARAM_INDEX: dict[ParamName, int] = {"v0": 0, "kappa": 1, "theta": 2, "xi": 3, "rho": 4}


@dataclass(frozen=True)
class ParamBounds:
    """Box bounds on the Heston parameters, passed straight to ``least_squares``."""

    lower: HestonParams
    upper: HestonParams

    def clip(self, params: HestonParams) -> HestonParams:
        """Clip ``params`` into the box (used to make a start feasible)."""
        lo = np.asarray(self.lower, dtype=np.float64)
        hi = np.asarray(self.upper, dtype=np.float64)
        return HestonParams.from_array(np.clip(np.asarray(params, dtype=np.float64), lo, hi))


#: Wide, economically sensible default bounds. ``v0``/``theta`` are variances
#: (vol up to ~100%); ``kappa`` spans slow-to-fast reversion; ``xi`` allows a
#: pronounced smile; ``rho`` is the full admissible correlation, held just off
#: :math:`\pm 1` so the characteristic function stays well-behaved.
DEFAULT_BOUNDS = ParamBounds(
    lower=HestonParams(v0=1e-6, kappa=1e-3, theta=1e-6, xi=1e-4, rho=-0.999),
    upper=HestonParams(v0=1.0, kappa=15.0, theta=1.0, xi=2.0, rho=0.999),
)


@dataclass(frozen=True)
class VolQuote:
    r"""One market vanilla quote: a strike, a maturity, and an implied volatility.

    Call and put implied vols coincide by put--call parity, so a quote is
    identified by ``(strike, expiry, implied_vol)`` alone; the calibrator picks
    the out-of-the-money option at that strike internally.

    Parameters
    ----------
    strike : float
        Strike ``K``. Must be positive.
    expiry : float
        Time to expiry ``T`` in years. Must be positive.
    implied_vol : float
        Observed Black--Scholes implied volatility. Must be non-negative.
    """

    strike: float
    expiry: float
    implied_vol: float

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.expiry <= 0.0:
            raise ValueError(f"expiry must be positive, got {self.expiry}")
        if self.implied_vol < 0.0:
            raise ValueError(f"implied_vol must be non-negative, got {self.implied_vol}")


class _FitQuality(NamedTuple):
    """Fit-quality summary at one parameter set."""

    rmse_vol: float
    rmse_price: float
    max_abs_vol: float
    model_ivs: FloatArray


@dataclass(frozen=True)
class HestonCalibrationResult:
    r"""Outcome of a Heston calibration — the fit plus its validation diagnostics.

    Attributes
    ----------
    params : HestonParams
        The calibrated parameters (lowest-cost start).
    process : HestonProcess
        The calibrated process (``params`` attached to the market).
    rmse_vol : float
        Root-mean-square fit error in **implied-vol points** across the quotes.
    rmse_price : float
        Root-mean-square fit error in **price**.
    max_abs_vol_error : float
        Largest absolute vol-point residual (the worst-fit quote).
    model_ivs : FloatArray
        Model implied vols at the quotes (aligned with the input order), for
        inspecting or plotting the fitted smile.
    feller_satisfied : bool
        Whether the calibrated parameters satisfy :math:`2\kappa\theta \ge \xi^2`.
    space : {"vol", "price"}
        The space the residuals were measured in.
    cost : float
        The optimizer's final cost (:math:`\tfrac12\sum r_i^2`) at the best start.
    success : bool
        Whether the best start's optimizer reported convergence.
    n_starts : int
        Number of starting points tried.
    n_quotes : int
        Number of market quotes fitted.
    message : str
        The optimizer's termination message for the best start.
    param_spread : HestonParams or None
        Per-parameter range (max minus min) across the *near-optimal* starts
        (those within 5% of the best cost). ``None`` for a single start. A small
        spread means the starts agree (the basin is found robustly); use
        :func:`profile_objective` to probe how flat the objective is along each
        axis, which is the sharper identifiability question.
    """

    params: HestonParams
    process: HestonProcess
    rmse_vol: float
    rmse_price: float
    max_abs_vol_error: float
    model_ivs: FloatArray
    feller_satisfied: bool
    space: CalibrationSpace
    cost: float
    success: bool
    n_starts: int
    n_quotes: int
    message: str
    param_spread: HestonParams | None

    def summary(self) -> str:
        """A compact multi-line report suitable for a script or the README."""
        p = self.params
        lines = [
            f"Heston calibration ({self.space} space, {self.n_quotes} quotes, "
            f"{self.n_starts} start(s))",
            f"  v0={p.v0:.4f}  kappa={p.kappa:.4f}  theta={p.theta:.4f}  "
            f"xi={p.xi:.4f}  rho={p.rho:+.4f}",
            f"  RMSE = {self.rmse_vol * 100:.3f} vol pts   "
            f"(max {self.max_abs_vol_error * 100:.3f})   "
            f"price RMSE = {self.rmse_price:.4e}",
            f"  Feller 2*kappa*theta >= xi^2: "
            f"{'satisfied' if self.feller_satisfied else 'VIOLATED'} "
            f"(2*k*t={2 * p.kappa * p.theta:.4f}, xi^2={p.xi**2:.4f})",
        ]
        if self.param_spread is not None:
            s = self.param_spread
            lines.append(
                f"  multi-start spread (near-optimal): "
                f"v0={s.v0:.3f} kappa={s.kappa:.3f} theta={s.theta:.3f} "
                f"xi={s.xi:.3f} rho={s.rho:.3f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ObjectiveProfile:
    """A one-parameter objective profile — the identifiability read-out.

    For each pinned value of one parameter, the other four are re-optimised and
    the minimised fit RMSE (in vol points) is recorded. A **flat** profile means
    the surface does not identify that parameter; a sharp ``V`` means it does.

    Attributes
    ----------
    param : str
        The profiled parameter name.
    values : FloatArray
        The pinned values, ascending.
    rmse_vol : FloatArray
        Minimised RMSE (vol points) at each pinned value.
    optimum : float
        The parameter value in ``values`` giving the lowest RMSE.
    flatness : float
        ``max(rmse_vol) - min(rmse_vol)`` over the scanned range — small means
        flat (weakly identified).
    """

    param: ParamName
    values: FloatArray
    rmse_vol: FloatArray
    optimum: float
    flatness: float


def vol_surface_from_grid(
    strikes: Sequence[float] | FloatArray,
    expiries: Sequence[float] | FloatArray,
    implied_vols: Sequence[Sequence[float]] | FloatArray,
) -> list[VolQuote]:
    """Build a list of :class:`VolQuote` from a rectangular strike/expiry grid.

    Parameters
    ----------
    strikes : sequence of float
        The ``n_k`` strikes (grid columns).
    expiries : sequence of float
        The ``n_t`` maturities (grid rows).
    implied_vols : 2-D array-like, shape ``(n_t, n_k)``
        ``implied_vols[i, j]`` is the implied vol at ``expiries[i]``, ``strikes[j]``.
    """
    k = np.asarray(strikes, dtype=np.float64)
    t = np.asarray(expiries, dtype=np.float64)
    iv = np.asarray(implied_vols, dtype=np.float64)
    if iv.shape != (t.size, k.size):
        raise ValueError(
            f"implied_vols must have shape (n_expiry, n_strike) = "
            f"({t.size}, {k.size}), got {iv.shape}"
        )
    return [
        VolQuote(strike=float(k[j]), expiry=float(t[i]), implied_vol=float(iv[i, j]))
        for i in range(t.size)
        for j in range(k.size)
    ]


# --------------------------------------------------------------------------- #
# Internal least-squares problem (shared by calibrate / profile / final RMSE).
# --------------------------------------------------------------------------- #


def _otm_option(market: Market, strike: float, expiry: float) -> EuropeanOption:
    """The out-of-the-money vanilla at ``(strike, expiry)`` (call above the forward)."""
    forward = market.forward(expiry)
    kind = OptionType.CALL if strike >= forward else OptionType.PUT
    return EuropeanOption(strike=strike, expiry=expiry, option_type=kind)


def _bs_price(market: Market, option: EuropeanOption, vol: float) -> float:
    """Black--Scholes price of ``option`` under ``market`` at volatility ``vol``."""
    return _ANALYTIC.calculate(option, BlackScholesProcess.from_market(market, vol))


def _model_iv(price: float, option: EuropeanOption, market: Market) -> float:
    """Invert a model ``price`` to Black--Scholes implied vol, guarded against the band.

    The FFT can return a marginally out-of-band price for a deep-OTM quote; we
    clamp into the no-arbitrage band so the objective stays finite and continuous
    rather than raising mid-optimization. Because we always invert the OTM option,
    the upper bound is the discounted spot (call) or discounted strike (put).
    """
    if not np.isfinite(price):
        # An extreme parameter corner overflowed the characteristic function;
        # signal "invalid" so the caller can penalise it rather than crash.
        return float("nan")
    T = option.expiry
    if option.option_type is OptionType.CALL:
        upper = market.spot * np.exp(-market.div * T)
    else:
        upper = option.strike * np.exp(-market.rate * T)
    clamped = min(max(price, _IV_PRICE_FLOOR), upper * _BAND_SHRINK)
    return implied_volatility(clamped, option, market)


@dataclass
class _Problem:
    """Pre-processed calibration problem: everything a residual evaluation needs."""

    market: Market
    options: list[EuropeanOption]
    market_ivs: FloatArray
    market_prices: FloatArray
    sqrt_w: FloatArray
    space: CalibrationSpace
    feller_weight: float
    engine: HestonFFTEngine

    @classmethod
    def build(
        cls,
        market: Market,
        quotes: Sequence[VolQuote],
        space: CalibrationSpace,
        weights: Sequence[float] | FloatArray | None,
        feller_weight: float,
        engine: HestonFFTEngine,
    ) -> _Problem:
        n = len(quotes)
        market_ivs = np.array([q.implied_vol for q in quotes], dtype=np.float64)
        options = [_otm_option(market, q.strike, q.expiry) for q in quotes]
        market_prices = np.array(
            [_bs_price(market, opt, iv) for opt, iv in zip(options, market_ivs, strict=True)],
            dtype=np.float64,
        )
        if weights is None:
            sqrt_w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError(f"weights must have length {n}, got shape {w.shape}")
            if np.any(w < 0.0):
                raise ValueError("weights must be non-negative")
            sqrt_w = np.sqrt(w)
        return cls(
            market=market,
            options=options,
            market_ivs=market_ivs,
            market_prices=market_prices,
            sqrt_w=sqrt_w,
            space=space,
            feller_weight=feller_weight,
            engine=engine,
        )

    def model_quantities(self, params: HestonParams) -> tuple[FloatArray, FloatArray]:
        """Model prices and model implied vols at every quote.

        Overflow/invalid warnings from an extreme parameter corner are silenced
        and surfaced as non-finite entries instead, which :meth:`residuals`
        converts into a large finite penalty.
        """
        process = params.to_process(self.market)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prices = np.array(
                [self.engine.calculate(opt, process) for opt in self.options], dtype=np.float64
            )
            ivs = np.array(
                [
                    _model_iv(float(p), opt, self.market)
                    for p, opt in zip(prices, self.options, strict=True)
                ],
                dtype=np.float64,
            )
        return prices, ivs

    def residuals(self, x: FloatArray) -> FloatArray:
        """Weighted residual vector for the full 5-parameter vector ``x``."""
        params = HestonParams.from_array(x)
        prices, ivs = self.model_quantities(params)
        base = (prices - self.market_prices) if self.space == "price" else (ivs - self.market_ivs)
        # A non-finite model price/IV means the pricer could not evaluate this
        # corner of the box; replace with a large finite penalty so least_squares
        # is repelled from it instead of failing on a NaN.
        base = np.where(np.isfinite(base), base, _BAD_RESIDUAL)
        res = self.sqrt_w * base
        if self.feller_weight > 0.0:
            penalty = np.sqrt(self.feller_weight) * max(
                0.0, params.xi**2 - 2.0 * params.kappa * params.theta
            )
            res = np.append(res, penalty)
        return res

    def quality(self, params: HestonParams) -> _FitQuality:
        """RMSE (vol and price) and model IVs at ``params``."""
        prices, model_ivs = self.model_quantities(params)
        vol_res = model_ivs - self.market_ivs
        price_res = prices - self.market_prices
        return _FitQuality(
            rmse_vol=float(np.sqrt(np.mean(vol_res**2))),
            rmse_price=float(np.sqrt(np.mean(price_res**2))),
            max_abs_vol=float(np.max(np.abs(vol_res))),
            model_ivs=model_ivs,
        )


def calibrate_heston(
    market: Market,
    quotes: Sequence[VolQuote],
    *,
    initial: HestonParams | None = None,
    bounds: ParamBounds = DEFAULT_BOUNDS,
    space: CalibrationSpace = "vol",
    weights: Sequence[float] | FloatArray | None = None,
    feller_weight: float = 0.0,
    n_starts: int = 1,
    rng: Generator | None = None,
    engine: HestonFFTEngine | None = None,
    max_nfev: int | None = None,
) -> HestonCalibrationResult:
    r"""Calibrate Heston parameters to a vanilla implied-volatility surface.

    Parameters
    ----------
    market : Market
        The market state (spot, rate, dividend) shared by every quote.
    quotes : sequence of VolQuote
        The market surface: strikes :math:`\times` maturities with implied vols.
    initial : HestonParams, optional
        Starting parameters for the first optimization. Defaults to a heuristic
        (``v0 = theta =`` mean market variance, ``kappa = 2``, ``xi = 0.5``,
        ``rho = -0.5``), clipped into ``bounds``.
    bounds : ParamBounds, optional
        Box bounds on the parameters (default :data:`DEFAULT_BOUNDS`).
    space : {"vol", "price"}, optional
        Residual space. ``"vol"`` (default) fits implied-vol points; ``"price"``
        fits option prices. See the module docstring for why vol space is the
        default.
    weights : sequence of float, optional
        Per-quote weights applied to the residuals (before squaring). Defaults to
        equal weight. Length must match ``quotes``.
    feller_weight : float, optional
        If positive, append a soft penalty
        :math:`\sqrt{w_F}\,\max(0,\ \xi^2 - 2\kappa\theta)` to the residuals,
        biasing the fit toward the Feller-satisfying region. Default ``0`` — the
        condition is *reported*, not enforced.
    n_starts : int, optional
        Number of starting points. ``> 1`` runs a multi-start (first start is
        ``initial``/heuristic, the rest are drawn uniformly within ``bounds``),
        keeps the lowest-cost fit, and reports the parameter spread across the
        near-optimal starts.
    rng : numpy.random.Generator, optional
        Seeded generator for the multi-start sampler (only used when
        ``n_starts > 1``). Defaults to ``default_rng(0)`` so results are
        deterministic; pass your own for a different seed.
    engine : HestonFFTEngine, optional
        The pricer used for model prices. Defaults to a fresh
        :class:`HestonFFTEngine`.
    max_nfev : int, optional
        Cap on residual evaluations per start (passed to ``least_squares``).

    Returns
    -------
    HestonCalibrationResult
        The calibrated parameters, fit quality (RMSE in vol points and price),
        the Feller flag, and the multi-start spread.
    """
    quotes = tuple(quotes)
    if not quotes:
        raise ValueError("need at least one quote to calibrate")
    if space not in ("vol", "price"):
        raise ValueError(f"space must be 'vol' or 'price', got {space!r}")
    if n_starts < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts}")
    if feller_weight < 0.0:
        raise ValueError(f"feller_weight must be non-negative, got {feller_weight}")

    engine = engine if engine is not None else HestonFFTEngine()
    problem = _Problem.build(market, quotes, space, weights, feller_weight, engine)

    lower = np.asarray(bounds.lower, dtype=np.float64)
    upper = np.asarray(bounds.upper, dtype=np.float64)
    starts = _starting_points(problem.market_ivs, initial, bounds, n_starts, rng)

    records: list[tuple[float, FloatArray]] = []
    best: tuple[float, FloatArray, bool, str] | None = None
    for x0 in starts:
        sol = least_squares(
            problem.residuals, x0, bounds=(lower, upper), x_scale="jac", max_nfev=max_nfev
        )
        cost = float(sol.cost)
        x = np.asarray(sol.x, dtype=np.float64)
        records.append((cost, x))
        if best is None or cost < best[0]:
            best = (cost, x, bool(sol.success), str(sol.message))

    assert best is not None  # at least one start always runs
    best_cost, best_x, best_success, best_message = best
    best_params = HestonParams.from_array(best_x)
    quality = problem.quality(best_params)

    return HestonCalibrationResult(
        params=best_params,
        process=best_params.to_process(market),
        rmse_vol=quality.rmse_vol,
        rmse_price=quality.rmse_price,
        max_abs_vol_error=quality.max_abs_vol,
        model_ivs=quality.model_ivs,
        feller_satisfied=best_params.feller_satisfied,
        space=space,
        cost=best_cost,
        success=best_success,
        n_starts=n_starts,
        n_quotes=len(quotes),
        message=best_message,
        param_spread=_param_spread(records, best_cost),
    )


def profile_objective(
    market: Market,
    quotes: Sequence[VolQuote],
    param: ParamName,
    values: Sequence[float] | FloatArray,
    *,
    anchor: HestonParams | None = None,
    bounds: ParamBounds = DEFAULT_BOUNDS,
    space: CalibrationSpace = "vol",
    weights: Sequence[float] | FloatArray | None = None,
    engine: HestonFFTEngine | None = None,
    max_nfev: int | None = None,
) -> ObjectiveProfile:
    r"""Profile the calibration objective along one parameter (identifiability).

    For each ``value`` the named ``param`` is *pinned* and the other four
    parameters are re-optimised; the minimised fit RMSE (in vol points) is
    returned. A flat profile means the surface barely constrains ``param`` (weakly
    identified, e.g. :math:`\kappa`); a sharp minimum means it does (e.g.
    :math:`\rho`).

    Parameters
    ----------
    market, quotes, bounds, space, weights, engine, max_nfev
        As in :func:`calibrate_heston`.
    param : {"v0", "kappa", "theta", "xi", "rho"}
        Which parameter to pin and scan.
    values : sequence of float
        The pinned values to evaluate (need not be sorted; the result is sorted).
    anchor : HestonParams, optional
        Starting point for the free parameters at each pin. Defaults to a single
        unconstrained :func:`calibrate_heston` fit, which makes the profiles
        smooth and cheap. Pass the known truth in a synthetic study.
    """
    if param not in _PARAM_INDEX:
        raise ValueError(f"param must be one of {list(_PARAM_INDEX)}, got {param!r}")
    engine = engine if engine is not None else HestonFFTEngine()
    problem = _Problem.build(market, tuple(quotes), space, weights, 0.0, engine)

    if anchor is None:
        anchor = calibrate_heston(
            market, quotes, bounds=bounds, space=space, weights=weights, engine=engine
        ).params

    idx = _PARAM_INDEX[param]
    free = [k for k in range(5) if k != idx]
    lower = np.asarray(bounds.lower, dtype=np.float64)
    upper = np.asarray(bounds.upper, dtype=np.float64)
    free_lo, free_hi = lower[free], upper[free]
    anchor_free = np.asarray(anchor, dtype=np.float64)[free]

    vals = np.sort(np.asarray(values, dtype=np.float64))
    rmse = np.empty(vals.size, dtype=np.float64)
    for m, value in enumerate(vals):
        pinned = float(np.clip(value, lower[idx], upper[idx]))

        def conditional(x_free: FloatArray, _pinned: float = pinned) -> FloatArray:
            x = np.empty(5, dtype=np.float64)
            x[idx] = _pinned
            x[free] = x_free
            return problem.residuals(x)

        sol = least_squares(
            conditional,
            anchor_free,
            bounds=(free_lo, free_hi),
            x_scale="jac",
            max_nfev=max_nfev,
        )
        x_best = np.empty(5, dtype=np.float64)
        x_best[idx] = pinned
        x_best[free] = np.asarray(sol.x, dtype=np.float64)
        rmse[m] = problem.quality(HestonParams.from_array(x_best)).rmse_vol

    return ObjectiveProfile(
        param=param,
        values=vals,
        rmse_vol=rmse,
        optimum=float(vals[int(np.argmin(rmse))]),
        flatness=float(rmse.max() - rmse.min()),
    )


def _starting_points(
    market_ivs: FloatArray,
    initial: HestonParams | None,
    bounds: ParamBounds,
    n_starts: int,
    rng: Generator | None,
) -> list[FloatArray]:
    """The list of feasible starting vectors: a heuristic first, then random draws."""
    if initial is None:
        atm_var = float(np.mean(market_ivs**2))
        initial = HestonParams(v0=atm_var, kappa=2.0, theta=atm_var, xi=0.5, rho=-0.5)
    first = np.asarray(bounds.clip(initial), dtype=np.float64)
    if n_starts == 1:
        return [first]

    generator = rng if rng is not None else np.random.default_rng(_DEFAULT_MULTISTART_SEED)
    lower = np.asarray(bounds.lower, dtype=np.float64)
    upper = np.asarray(bounds.upper, dtype=np.float64)
    extra = [lower + generator.random(lower.size) * (upper - lower) for _ in range(n_starts - 1)]
    return [first, *extra]


def _param_spread(
    records: Sequence[tuple[float, FloatArray]], best_cost: float
) -> HestonParams | None:
    """Per-parameter range across starts within ``_SPREAD_REL_TOL`` of the best cost.

    A wide range means several near-equal-cost optima disagree on that parameter
    (the objective is flat along it). ``None`` when only one start was near-optimal.
    """
    threshold = best_cost * (1.0 + _SPREAD_REL_TOL) + 1e-30
    near = np.array([x for cost, x in records if cost <= threshold], dtype=np.float64)
    if near.shape[0] < 2:
        return None
    return HestonParams.from_array(near.max(axis=0) - near.min(axis=0))
