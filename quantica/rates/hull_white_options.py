r"""Hull--White option pricing and volatility calibration — closing the rates loop.

Where :mod:`quantica.rates.products` prices caps/swaptions with market-standard Black-76, this
module prices the *same* instruments **analytically under Hull--White**, and calibrates the
model's volatility to the market — the step that finally identifies :math:`\sigma`, which the
curve alone could not (step 2 found the curve pins :math:`\sigma` only through a tiny convexity
term).

The machinery is the affine **zero-coupon bond option** (:func:`hull_white_bond_option`), from
which everything else follows:

* a **caplet** is a put on a discount bond — :math:`\tau L(T_r,T_p)` reset is
  :math:`(1+K\tau)` puts on the :math:`T_p`-bond struck at :math:`1/(1+K\tau)`;
* a **European swaption** decomposes by **Jamshidian** into a portfolio of bond options: in a
  one-factor model the coupon bond is monotone in the short rate, so a single :math:`r^\*`
  splits the payoff into per-cashflow bond options struck at :math:`P(r^\*; T_0, T_i)`.

Each analytic price is cross-checked against a Hull--White Monte Carlo estimate (exact-transition
simulation), and against Black-76 through the implied volatility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from quantica.rates.products import Cap, Caplet, Swaption
from quantica.rates.short_rate import HullWhite

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "HullWhiteVolResult",
    "calibrate_hull_white_volatility",
    "hull_white_bond_option",
    "hull_white_cap",
    "hull_white_caplet",
    "hull_white_price",
    "hull_white_price_mc",
    "hull_white_swaption",
]


def hull_white_bond_option(
    model: HullWhite,
    strike: float,
    expiry: float,
    bond_maturity: float,
    *,
    call: bool,
) -> float:
    r"""Analytic Hull--White price of a European option on a zero-coupon bond.

    Prices an option expiring at ``expiry`` on the ``bond_maturity``-bond, struck at ``strike``.

    Parameters
    ----------
    model : HullWhite
        The fitted Hull--White model.
    strike : float
        Option strike (a bond price in ``(0, 1]``).
    expiry : float
        Option expiry ``T_O`` in years.
    bond_maturity : float
        Underlying bond maturity ``T_B > T_O``.
    call : bool
        ``True`` for a call on the bond, ``False`` for a put.

    Returns
    -------
    float
        The option present value.
    """
    a, sigma = model.a, model.sigma
    p_expiry = float(model.curve.discount_factor(expiry))
    p_bond = float(model.curve.discount_factor(bond_maturity))
    big_b = (1.0 - np.exp(-a * (bond_maturity - expiry))) / a
    sigma_p = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * expiry)) / (2.0 * a)) * big_b
    if sigma_p <= 0.0:  # no volatility -> intrinsic value on the forward bond
        omega = 1.0 if call else -1.0
        return float(max(omega * (p_bond - strike * p_expiry), 0.0))
    h = np.log(p_bond / (strike * p_expiry)) / sigma_p + 0.5 * sigma_p
    if call:
        return float(p_bond * norm.cdf(h) - strike * p_expiry * norm.cdf(h - sigma_p))
    return float(strike * p_expiry * norm.cdf(-h + sigma_p) - p_bond * norm.cdf(-h))


def hull_white_caplet(model: HullWhite, caplet: Caplet) -> float:
    r"""Analytic Hull--White caplet/floorlet price via the put-on-bond equivalence."""
    tau, strike = caplet.accrual, caplet.strike
    bond_strike = 1.0 / (1.0 + strike * tau)
    # A caplet is (1+Kτ) puts on the payment-date bond; a floorlet, (1+Kτ) calls.
    return float(
        (1.0 + strike * tau)
        * hull_white_bond_option(
            model, bond_strike, caplet.reset, caplet.payment, call=caplet.is_floor
        )
    )


def hull_white_cap(model: HullWhite, cap: Cap) -> float:
    """Analytic Hull--White cap/floor price: the sum over the caplet/floorlet strip."""
    return float(sum(hull_white_caplet(model, c) for c in cap.caplets))


def hull_white_swaption(model: HullWhite, swaption: Swaption) -> float:
    r"""Analytic Hull--White swaption price via the Jamshidian decomposition.

    Finds the critical short rate :math:`r^\*` at which the underlying coupon bond equals par,
    then values the swaption as the portfolio of per-cashflow bond options struck at the bond
    prices :math:`P(r^\*; T_0, T_i)` (a payer swaption is a portfolio of *puts*, a receiver a
    portfolio of *calls*).
    """
    expiry = swaption.expiry
    times = swaption.payment_times
    tau = 1.0 / swaption.frequency
    cashflows = np.full(times.size, swaption.strike * tau)
    cashflows[-1] += 1.0  # add the notional at the final payment

    def coupon_bond(rate: float) -> float:
        bonds = np.array([float(model.zero_coupon_bond(expiry, t, rate)) for t in times])
        return float(np.sum(cashflows * bonds) - 1.0)

    r_star = brentq(coupon_bond, -1.0, 1.0, xtol=1e-14)
    strikes = np.array([float(model.zero_coupon_bond(expiry, t, r_star)) for t in times])
    # Payer -> option to pay fixed -> puts on the bonds; receiver -> calls.
    call = not swaption.payer
    return float(
        sum(
            c * hull_white_bond_option(model, k, expiry, t, call=call)
            for c, k, t in zip(cashflows, strikes, times, strict=True)
        )
    )


def hull_white_price(model: HullWhite, product: Caplet | Cap | Swaption) -> float:
    """Analytic Hull--White price dispatched on the product type."""
    if isinstance(product, Caplet):
        return hull_white_caplet(model, product)
    if isinstance(product, Cap):
        return hull_white_cap(model, product)
    if isinstance(product, Swaption):
        return hull_white_swaption(model, product)
    raise TypeError(f"unsupported product type {type(product).__name__}")


def hull_white_price_mc(
    model: HullWhite,
    product: Caplet | Swaption,
    *,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    r"""Hull--White Monte Carlo price of a caplet or swaption, with its standard error.

    Simulates the short rate to the option expiry (exact transitions), discounting each path by
    the trapezoidal integral of the rate — the cross-check for the analytic formulas.

    Parameters
    ----------
    model : HullWhite
        The Hull--White model.
    product : Caplet or Swaption
        The instrument to price.
    n_paths, n_steps : int
        Number of paths and time steps to the expiry.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    tuple of float
        ``(price_estimate, standard_error)``.
    """
    expiry = product.reset if isinstance(product, Caplet) else product.expiry
    grid = np.linspace(0.0, expiry, n_steps + 1)
    paths = model.simulate(grid, n_paths, rng=rng)
    discount = np.exp(-np.trapezoid(paths, grid, axis=1))  # D(0, expiry)
    rate_at_expiry = paths[:, -1]

    if isinstance(product, Caplet):
        bond = np.asarray(model.zero_coupon_bond(product.reset, product.payment, rate_at_expiry))
        libor = (1.0 / bond - 1.0) / product.accrual
        omega = -1.0 if product.is_floor else 1.0
        payoff = product.accrual * np.maximum(omega * (libor - product.strike), 0.0)
        payoff = payoff * bond  # discount from payment back to expiry
    else:
        payoff = _swaption_payoff(model, product, rate_at_expiry)

    values = discount * payoff
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(n_paths))


def _swaption_payoff(
    model: HullWhite, swaption: Swaption, rate_at_expiry: FloatArray
) -> FloatArray:
    """Per-path swaption payoff at expiry: ``max(omega*(swap value), 0)``."""
    times = swaption.payment_times
    tau = 1.0 / swaption.frequency
    cashflows = np.full(times.size, swaption.strike * tau)
    cashflows[-1] += 1.0
    coupon_bond = np.zeros_like(rate_at_expiry)
    for c, t in zip(cashflows, times, strict=True):
        coupon_bond = coupon_bond + c * np.asarray(
            model.zero_coupon_bond(swaption.expiry, t, rate_at_expiry)
        )
    receive_fixed = coupon_bond - 1.0
    omega = -1.0 if swaption.payer else 1.0  # payer profits when the fixed leg is cheap
    return np.asarray(np.maximum(omega * receive_fixed, 0.0), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Volatility calibration — what finally identifies sigma
# --------------------------------------------------------------------------- #


class HullWhiteVolResult:
    """The Hull--White volatility calibrated to a set of option quotes.

    Attributes
    ----------
    model : HullWhite
        The calibrated model (fitted ``sigma``, given ``a``).
    sigma : float
        The fitted volatility.
    rmse : float
        Root-mean-square price error across the calibration quotes.
    """

    def __init__(self, model: HullWhite, sigma: float, rmse: float) -> None:
        self.model = model
        self.sigma = sigma
        self.rmse = rmse


def calibrate_hull_white_volatility(
    curve: object,
    quotes: list[tuple[Caplet | Cap | Swaption, float]],
    *,
    a: float,
    sigma_bounds: tuple[float, float] = (1e-5, 0.2),
) -> HullWhiteVolResult:
    r"""Calibrate Hull--White's :math:`\sigma` to market option prices (``a`` given).

    Unlike the curve (which identifies :math:`\sigma` only through a tiny convexity term, so
    Hull--White reprices it for *any* :math:`\sigma`), option prices are strongly volatility
    dependent, so this least-squares fit pins :math:`\sigma` down.

    Parameters
    ----------
    curve : DiscountCurve
        The discount curve the model is built on.
    quotes : list of (product, price)
        Market instruments (caplets / caps / swaptions) and their target prices.
    a : float
        The (given) mean-reversion speed.
    sigma_bounds : tuple of float, optional
        Bounds for the fitted volatility (default ``(1e-5, 0.2)``).

    Returns
    -------
    HullWhiteVolResult
        The calibrated model, fitted :math:`\sigma`, and the fit RMSE.
    """
    if not quotes:
        raise ValueError("need at least one quote to calibrate")

    def residuals(params: FloatArray) -> FloatArray:
        model = HullWhite.from_curve(curve, a=a, sigma=float(params[0]))  # type: ignore[arg-type]
        return np.array(
            [hull_white_price(model, product) - price for product, price in quotes],
            dtype=np.float64,
        )

    fit = least_squares(residuals, [0.01], bounds=([sigma_bounds[0]], [sigma_bounds[1]]))
    sigma = float(fit.x[0])
    model = HullWhite.from_curve(curve, a=a, sigma=sigma)  # type: ignore[arg-type]
    rmse = float(np.sqrt(np.mean(residuals(fit.x) ** 2)))
    return HullWhiteVolResult(model=model, sigma=sigma, rmse=rmse)
