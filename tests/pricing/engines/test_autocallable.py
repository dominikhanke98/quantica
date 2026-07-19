"""Validation of the autocallable note (numerical-validation skill).

Four kinds of check, matching the "compose validated pieces" claim:

* **Path simulators anchored to their pricers** — the new Merton and Heston full-path
  simulators reprice a European call to within Monte Carlo error of the closed-form /
  FFT engines they are meant to be consistent with.
* **Payoff wiring pinned by static replication** — a single-observation note decomposes
  into cash-or-nothing digitals plus an asset-or-nothing put, priced in closed form under
  Black--Scholes; the Monte Carlo engine must reproduce it.
* **Structural limits** — autocall barrier -> 0 collapses to a one-period coupon+principal
  payment; full downside protection removes all loss; the autocall/maturity probabilities
  partition to one.
* **The headline** — a flat-vol Black--Scholes price, matched to Heston's ATM implied vol,
  *overprices* the note versus the skew-consistent Heston price, because it ignores the
  negative skew that makes the embedded short down-and-in put dearer.
"""

from __future__ import annotations

import numpy as np
from quantica.pricing import (
    AutocallableMonteCarloEngine,
    AutocallableNote,
    BlackScholesProcess,
    EuropeanOption,
    HestonFFTEngine,
    HestonProcess,
    Market,
    MertonClosedFormEngine,
    MertonProcess,
    OptionType,
    implied_volatility,
)
from quantica.pricing.engines._paths import HestonPathSimulator, MertonPathSimulator
from scipy.stats import norm

# --------------------------------------------------------------------------- #
# Path simulators anchored to their transform / closed-form pricers
# --------------------------------------------------------------------------- #


def _european_call_mc(paths: np.ndarray, strike: float, rate: float, expiry: float) -> tuple:
    disc = np.exp(-rate * expiry)
    payoff = disc * np.maximum(paths[:, -1] - strike, 0.0)
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(payoff.shape[0]))


def test_merton_path_simulator_matches_closed_form() -> None:
    """Merton full-path MC reprices a European call to within MC error of the closed form."""
    spot, rate, strike, expiry = 100.0, 0.03, 100.0, 1.0
    process = MertonProcess(spot=spot, rate=rate, vol=0.2, lam=0.5, mu_j=-0.1, sigma_j=0.15)
    sim = MertonPathSimulator(200_000, rng=np.random.default_rng(0))
    paths = sim.simulate(
        spot=spot,
        rate=rate,
        div=0.0,
        vol=0.2,
        lam=0.5,
        mu_j=-0.1,
        sigma_j=0.15,
        dt=expiry,
        n_steps=1,
    )
    mc, se = _european_call_mc(paths, strike, rate, expiry)
    option = EuropeanOption(strike=strike, expiry=expiry, option_type=OptionType.CALL)
    reference = MertonClosedFormEngine().calculate(option, process)
    assert abs(mc - reference) < 4.0 * se


def test_heston_path_simulator_matches_fft() -> None:
    """Heston full-truncation MC reprices a European call to within MC error of the FFT."""
    spot, rate, strike, expiry = 100.0, 0.03, 100.0, 1.0
    process = HestonProcess(spot=spot, rate=rate, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)
    sim = HestonPathSimulator(120_000, rng=np.random.default_rng(1), n_substeps=64)
    paths = sim.simulate(
        spot=spot,
        rate=rate,
        div=0.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        dt=expiry,
        n_steps=1,
    )
    mc, se = _european_call_mc(paths, strike, rate, expiry)
    option = EuropeanOption(strike=strike, expiry=expiry, option_type=OptionType.CALL)
    reference = HestonFFTEngine().calculate(option, process)
    assert abs(mc - reference) < 4.0 * se


# --------------------------------------------------------------------------- #
# Payoff wiring pinned by static replication (single observation, Black--Scholes)
# --------------------------------------------------------------------------- #


def test_single_observation_matches_static_replication() -> None:
    """A one-observation note = digitals + asset-or-nothing put, priced in closed form.

    At maturity the note pays ``N(1+c)`` above the autocall barrier, ``N`` between the two
    barriers, and ``N * S_T/S_0`` below the downside barrier. Under Black--Scholes each
    region is a European digital / asset-or-nothing claim, so the whole note has a closed
    form that the Monte Carlo engine must reproduce.
    """
    spot, rate, vol, expiry = 100.0, 0.03, 0.2, 2.0
    notional, coupon, auto, down = 1.0, 0.05, 1.0, 0.6
    process = BlackScholesProcess(spot=spot, rate=rate, vol=vol)
    note = AutocallableNote(
        maturity=expiry,
        n_observations=1,
        autocall_barrier=auto,
        coupon=coupon,
        downside_barrier=down,
    )

    disc = np.exp(-rate * expiry)
    srt = vol * np.sqrt(expiry)

    def d2(level: float) -> float:
        return float((np.log(spot / level) + (rate - 0.5 * vol * vol) * expiry) / srt)

    def d1(level: float) -> float:
        return d2(level) + srt

    auto_lvl, down_lvl = auto * spot, down * spot
    # Above autocall: cash notional*(1+coupon).
    above = notional * (1.0 + coupon) * disc * norm.cdf(d2(auto_lvl))
    # Between barriers: cash notional.
    between = notional * disc * (norm.cdf(d2(down_lvl)) - norm.cdf(d2(auto_lvl)))
    # Below downside: notional * S_T / S_0 -> asset-or-nothing put (div = 0).
    below = notional * norm.cdf(-d1(down_lvl))
    reference = above + between + below

    result = AutocallableMonteCarloEngine(300_000, rng=np.random.default_rng(3)).estimate(
        note, process
    )
    assert abs(result.price - reference) < 4.0 * result.std_error


# --------------------------------------------------------------------------- #
# Structural limits
# --------------------------------------------------------------------------- #


def test_autocall_barrier_to_zero_is_one_period_coupon_bond() -> None:
    """With the autocall barrier ~ 0 every path redeems at the first date: a fixed cashflow."""
    spot, rate = 100.0, 0.03
    process = BlackScholesProcess(spot=spot, rate=rate, vol=0.25)
    note = AutocallableNote(
        maturity=3.0, n_observations=3, autocall_barrier=1e-9, coupon=0.06, downside_barrier=0.6
    )
    result = AutocallableMonteCarloEngine(50_000, rng=np.random.default_rng(4)).estimate(
        note, process
    )
    expected = (1.0 + 0.06) * np.exp(-rate * 1.0)  # notional*(1+coupon) discounted to t_1
    assert np.isclose(result.price, expected, atol=1e-12)  # deterministic: no dispersion
    assert result.std_error < 1e-15  # every path pays the same fixed cashflow
    assert result.autocall_probabilities[0] == 1.0


def test_full_downside_protection_removes_loss() -> None:
    """A zero downside barrier is principal-protected: no loss, price above the principal PV."""
    spot, rate, maturity = 100.0, 0.03, 3.0
    process = BlackScholesProcess(spot=spot, rate=rate, vol=0.2)
    note = AutocallableNote(
        maturity=maturity, n_observations=3, autocall_barrier=1.0, coupon=0.05, downside_barrier=0.0
    )
    result = AutocallableMonteCarloEngine(80_000, rng=np.random.default_rng(5)).estimate(
        note, process
    )
    assert result.loss_probability == 0.0
    assert result.price >= np.exp(-rate * maturity)  # >= PV of principal


def test_autocall_and_maturity_probabilities_partition() -> None:
    """Every path either autocalls once or survives to maturity — the probabilities sum to one."""
    process = BlackScholesProcess(spot=100.0, rate=0.03, vol=0.22)
    note = AutocallableNote(
        maturity=4.0, n_observations=8, autocall_barrier=1.0, coupon=0.03, downside_barrier=0.65
    )
    result = AutocallableMonteCarloEngine(60_000, rng=np.random.default_rng(6)).estimate(
        note, process
    )
    total = float(result.autocall_probabilities.sum()) + result.maturity_probability
    assert np.isclose(total, 1.0, atol=1e-12)
    assert result.loss_probability <= result.maturity_probability


# --------------------------------------------------------------------------- #
# The headline: flat vol misprices a skew-sensitive structured product
# --------------------------------------------------------------------------- #


def test_flat_vol_overprices_versus_heston_skew() -> None:
    """Black--Scholes matched to Heston's ATM vol *overprices* the note (it ignores skew).

    The holder is short a down-and-in put — short skew. Matching the flat Black--Scholes
    vol to Heston's at-the-money implied vol isolates the skew: the steep negative skew
    makes the embedded put dearer, so the skew-consistent Heston price sits materially
    *below* the flat-vol price, with a higher probability of capital loss.
    """
    spot, rate, maturity = 100.0, 0.03, 3.0
    heston = HestonProcess(spot=spot, rate=rate, v0=0.05, kappa=1.5, theta=0.05, xi=0.9, rho=-0.8)
    atm = EuropeanOption(strike=spot, expiry=maturity, option_type=OptionType.CALL)
    atm_iv = implied_volatility(
        HestonFFTEngine().calculate(atm, heston), atm, Market(spot=spot, rate=rate)
    )
    flat_bs = BlackScholesProcess(spot=spot, rate=rate, vol=atm_iv)

    note = AutocallableNote(
        maturity=maturity, n_observations=6, autocall_barrier=1.0, coupon=0.04, downside_barrier=0.6
    )
    bs = AutocallableMonteCarloEngine(300_000, rng=np.random.default_rng(7)).estimate(note, flat_bs)
    hn = AutocallableMonteCarloEngine(
        300_000, rng=np.random.default_rng(7), heston_substeps=64
    ).estimate(note, heston)

    gap = bs.price - hn.price
    combined_se = float(np.hypot(bs.std_error, hn.std_error))
    assert gap > 8.0 * combined_se  # material and correctly signed
    assert gap > 0.004  # > 0.4% of notional
    assert hn.loss_probability > bs.loss_probability  # skew fattens the left tail


# --------------------------------------------------------------------------- #
# Reproducibility and the engine contract
# --------------------------------------------------------------------------- #


def test_seeded_runs_are_reproducible() -> None:
    """The injected generator makes the price bit-for-bit reproducible; SE governs seed spread."""
    process = BlackScholesProcess(spot=100.0, rate=0.03, vol=0.2)
    note = AutocallableNote(
        maturity=3.0, n_observations=3, autocall_barrier=1.0, coupon=0.05, downside_barrier=0.6
    )
    a = AutocallableMonteCarloEngine(40_000, rng=np.random.default_rng(11)).estimate(note, process)
    b = AutocallableMonteCarloEngine(40_000, rng=np.random.default_rng(11)).estimate(note, process)
    c = AutocallableMonteCarloEngine(40_000, rng=np.random.default_rng(12)).estimate(note, process)
    assert a.price == b.price  # same seed -> identical
    assert abs(a.price - c.price) < 4.0 * np.hypot(a.std_error, c.std_error)  # seeds agree


def test_rejects_wrong_instrument() -> None:
    """The engine prices an AutocallableNote, not a plain option."""
    engine = AutocallableMonteCarloEngine(1_000, rng=np.random.default_rng(0))
    option = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    process = BlackScholesProcess(spot=100.0, rate=0.03, vol=0.2)
    try:
        engine.estimate(option, process)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "AutocallableNote" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected TypeError for a non-autocallable instrument")


def test_engine_satisfies_price_only_contract() -> None:
    """`calculate` returns the same point estimate as `estimate` under a shared seed."""
    process = BlackScholesProcess(spot=100.0, rate=0.03, vol=0.2)
    note = AutocallableNote(
        maturity=2.0, n_observations=4, autocall_barrier=1.0, coupon=0.04, downside_barrier=0.6
    )
    price = AutocallableMonteCarloEngine(20_000, rng=np.random.default_rng(9)).calculate(
        note, process
    )
    full = AutocallableMonteCarloEngine(20_000, rng=np.random.default_rng(9)).estimate(
        note, process
    )
    assert price == full.price
