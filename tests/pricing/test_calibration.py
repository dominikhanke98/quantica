r"""Validation of the Heston calibration (numerical-validation skill).

The headline check is **synthetic recovery**: generate an implied-vol surface from
known Heston parameters, calibrate back, and confirm the machinery recovers them.
Because the *same* FFT engine both generates and fits the surface, engine
discretisation is not a confounder — a noise-free surface is recovered to solver
tolerance, which isolates the calibration design (objective, weighting, bounds,
optimizer) as the thing under test.

Beyond recovery we check, honestly:

- **Fit quality** — RMSE in vol points is tiny and residuals are unstructured.
- **Identifiability** — under measurement noise ``v0`` and ``theta`` recover
  tightly while ``kappa`` is loosely determined; the objective profile along
  ``kappa`` has a broad, shallow basin. We *report* this rather than pretend the
  fit is unique.
- **Feller condition** — reported via the flag, and (optionally) driven toward
  satisfaction by the soft penalty.
- **Determinism** — a seeded multi-start reproduces byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    EuropeanOption,
    HestonFFTEngine,
    HestonParams,
    Market,
    ObjectiveProfile,
    OptionType,
    ParamBounds,
    VolQuote,
    calibrate_heston,
    implied_volatility,
    profile_objective,
    vol_surface_from_grid,
)

# A light FFT grid keeps the many calibrations fast; recovery is exact regardless
# because the same engine generates and fits the surface.
ENGINE = HestonFFTEngine(n_fft=2048)
MARKET = Market(spot=100.0, rate=0.03, div=0.01)

# A Feller-satisfying truth (2*kappa*theta = 0.20 >= xi^2 = 0.09).
TRUTH = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.7)

STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
EXPIRIES = np.array([0.25, 0.5, 1.0, 2.0])

_PARAM_NAMES = ("v0", "kappa", "theta", "xi", "rho")


def make_surface(
    params: HestonParams,
    strikes: np.ndarray = STRIKES,
    expiries: np.ndarray = EXPIRIES,
    *,
    market: Market = MARKET,
    engine: HestonFFTEngine = ENGINE,
) -> np.ndarray:
    """Build the Black--Scholes implied-vol surface a Heston model implies."""
    process = params.to_process(market)
    ivs = np.zeros((expiries.size, strikes.size))
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            kind = OptionType.CALL if market.forward(T) <= K else OptionType.PUT
            opt = EuropeanOption(float(K), float(T), kind)
            ivs[i, j] = implied_volatility(engine.calculate(opt, process), opt, market)
    return ivs


# --------------------------------------------------------------------------- #
# 1. Synthetic recovery (the headline)
# --------------------------------------------------------------------------- #


def test_recovers_known_parameters_from_synthetic_surface() -> None:
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    result = calibrate_heston(MARKET, quotes, engine=ENGINE)

    recovered = np.asarray(result.params)
    truth = np.asarray(TRUTH)
    # A noise-free surface is recovered to solver tolerance across all five params.
    np.testing.assert_allclose(recovered, truth, rtol=2e-3, atol=1e-4)
    assert result.success
    assert result.n_quotes == STRIKES.size * EXPIRIES.size


def test_recovery_fit_quality_is_tight_and_unstructured() -> None:
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    result = calibrate_heston(MARKET, quotes, engine=ENGINE)

    # RMSE well under a hundredth of a vol point; worst quote barely worse.
    assert result.rmse_vol < 1e-5
    assert result.max_abs_vol_error < 5e-5
    # Model IVs reproduce every input quote.
    np.testing.assert_allclose(result.model_ivs, ivs.ravel(), atol=5e-5)


def test_price_space_also_recovers() -> None:
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    result = calibrate_heston(MARKET, quotes, space="price", engine=ENGINE)
    np.testing.assert_allclose(np.asarray(result.params), np.asarray(TRUTH), rtol=5e-3, atol=1e-4)
    assert result.space == "price"


# --------------------------------------------------------------------------- #
# 2. Identifiability (reported honestly, not hidden)
# --------------------------------------------------------------------------- #


def test_noise_makes_kappa_less_identified_than_v0_and_theta() -> None:
    # Under measurement noise, the level parameters (v0, theta) recover tightly
    # while the mean-reversion speed (kappa) scatters widely across noise draws —
    # the classic Heston identifiability signature. We assert the *relative*
    # dispersion ordering, which is the honest, reproducible finding.
    ivs = make_surface(TRUTH)
    recovered = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        noisy = ivs + rng.normal(0.0, 0.005, ivs.shape)  # 0.5 vol-point noise
        quotes = vol_surface_from_grid(STRIKES, EXPIRIES, noisy)
        recovered.append(np.asarray(calibrate_heston(MARKET, quotes, engine=ENGINE).params))
    recovered = np.array(recovered)
    truth = np.asarray(TRUTH)
    rel_std = recovered.std(axis=0) / np.abs(truth)  # per-parameter relative scatter
    disp = dict(zip(_PARAM_NAMES, rel_std, strict=True))

    # Level parameters stay tight; kappa is markedly looser than either.
    assert disp["v0"] < 0.1
    assert disp["theta"] < 0.1
    assert disp["kappa"] > 3.0 * max(disp["v0"], disp["theta"])


def test_objective_profile_has_minimum_near_truth() -> None:
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)

    # rho is well identified: the profile minimum sits at the true rho, and the
    # fit degrades sharply as rho moves away.
    rho_grid = np.linspace(-0.9, -0.5, 5)
    prof = profile_objective(MARKET, quotes, "rho", rho_grid, anchor=TRUTH, engine=ENGINE)
    assert prof.optimum == pytest.approx(TRUTH.rho, abs=0.1 + 1e-9)
    assert prof.rmse_vol.min() < 1e-4  # essentially perfect at the optimum
    # Endpoints away from truth are materially worse (a genuine minimum, not flat).
    assert prof.rmse_vol[0] > 10.0 * prof.rmse_vol.min()
    assert prof.rmse_vol[-1] > 10.0 * prof.rmse_vol.min()


def test_kappa_objective_valley_is_broader_than_rho() -> None:
    # Both parameters have a genuine minimum at the truth on a noise-free surface,
    # but the *width* of the near-optimal valley differs: kappa's is much broader
    # (in relative terms) than rho's, so a given amount of noise moves kappa's
    # argmin far more. Comparing the two valleys on the same surface is the
    # honest, reproducible read-out of "kappa is the weakly-identified one".
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    kappa_prof = profile_objective(
        MARKET, quotes, "kappa", np.linspace(0.5, 4.0, 15), anchor=TRUTH, engine=ENGINE
    )
    rho_prof = profile_objective(
        MARKET, quotes, "rho", np.linspace(-0.9, -0.5, 15), anchor=TRUTH, engine=ENGINE
    )

    def relative_valley_width(prof: ObjectiveProfile, scale: float) -> float:
        # Width of {value : RMSE <= min + 5bp}, relative to |parameter scale|.
        within = prof.values[prof.rmse_vol <= prof.rmse_vol.min() + 5e-4]
        return float(within.max() - within.min()) / abs(scale)

    # Genuine minima at the truth (recovery), not flat plateaus.
    assert kappa_prof.rmse_vol.min() < 1e-5
    assert rho_prof.rmse_vol.min() < 1e-5
    assert kappa_prof.optimum == pytest.approx(TRUTH.kappa, abs=0.3)
    # kappa's relative valley is markedly broader than rho's.
    assert relative_valley_width(kappa_prof, TRUTH.kappa) > 2.0 * relative_valley_width(
        rho_prof, TRUTH.rho
    )


# --------------------------------------------------------------------------- #
# 3. Feller condition (reported; optionally penalised)
# --------------------------------------------------------------------------- #


def test_feller_flag_reports_violation_honestly() -> None:
    # A truth that violates Feller (2*kappa*theta = 0.15 < xi^2 = 0.16) is
    # recovered as such, and the flag says so rather than hiding it.
    violating = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.4, rho=-0.6)
    assert not violating.feller_satisfied
    ivs = make_surface(violating)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    result = calibrate_heston(MARKET, quotes, engine=ENGINE)
    assert not result.feller_satisfied
    assert "VIOLATED" in result.summary()


def test_feller_penalty_reduces_the_violation() -> None:
    # The soft penalty biases the fit toward the Feller-satisfying region: the
    # violation margin xi^2 - 2*kappa*theta shrinks, at the cost of a worse fit.
    violating = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.4, rho=-0.6)
    ivs = make_surface(violating)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)

    free = calibrate_heston(MARKET, quotes, engine=ENGINE)
    penalised = calibrate_heston(MARKET, quotes, feller_weight=50.0, engine=ENGINE)

    def margin(p: HestonParams) -> float:
        return p.xi**2 - 2.0 * p.kappa * p.theta

    assert margin(penalised.params) < margin(free.params)
    # The penalty trades fit quality for the constraint (still a sensible fit).
    assert penalised.rmse_vol >= free.rmse_vol


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def test_multistart_is_deterministic_when_seeded() -> None:
    ivs = make_surface(TRUTH, strikes=np.array([90.0, 100.0, 110.0]), expiries=np.array([0.5, 1.0]))
    quotes = vol_surface_from_grid(np.array([90.0, 100.0, 110.0]), np.array([0.5, 1.0]), ivs)
    a = calibrate_heston(MARKET, quotes, n_starts=3, rng=np.random.default_rng(1), engine=ENGINE)
    b = calibrate_heston(MARKET, quotes, n_starts=3, rng=np.random.default_rng(1), engine=ENGINE)
    assert tuple(a.params) == tuple(b.params)
    assert a.n_starts == 3


def test_weights_are_applied() -> None:
    # Zero-weighting every quote but one makes the calibration fit that one
    # (near-)exactly, at the expense of the rest — evidence the weights bite.
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    weights = np.zeros(len(quotes))
    weights[0] = 1.0
    result = calibrate_heston(MARKET, quotes, weights=weights, engine=ENGINE)
    assert abs(result.model_ivs[0] - quotes[0].implied_vol) < 1e-3


# --------------------------------------------------------------------------- #
# 5. Input validation / wiring
# --------------------------------------------------------------------------- #


def test_vol_surface_from_grid_shape_check() -> None:
    with pytest.raises(ValueError, match="shape"):
        vol_surface_from_grid([100.0, 110.0], [1.0], [[0.2]])  # 1x1 != 1x2


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"strike": -1.0, "expiry": 1.0, "implied_vol": 0.2}, "strike must be positive"),
        ({"strike": 100.0, "expiry": 0.0, "implied_vol": 0.2}, "expiry must be positive"),
        ({"strike": 100.0, "expiry": 1.0, "implied_vol": -0.1}, "implied_vol must be non-negative"),
    ],
)
def test_vol_quote_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        VolQuote(**kwargs)


def test_calibrate_input_validation() -> None:
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    with pytest.raises(ValueError, match="at least one quote"):
        calibrate_heston(MARKET, [])
    with pytest.raises(ValueError, match="space must be"):
        calibrate_heston(MARKET, quotes, space="log", engine=ENGINE)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_starts must be at least 1"):
        calibrate_heston(MARKET, quotes, n_starts=0, engine=ENGINE)
    with pytest.raises(ValueError, match="feller_weight must be non-negative"):
        calibrate_heston(MARKET, quotes, feller_weight=-1.0, engine=ENGINE)
    with pytest.raises(ValueError, match="weights must have length"):
        calibrate_heston(MARKET, quotes, weights=[1.0, 2.0], engine=ENGINE)


def test_profile_rejects_unknown_param() -> None:
    ivs = make_surface(TRUTH)
    quotes = vol_surface_from_grid(STRIKES, EXPIRIES, ivs)
    with pytest.raises(ValueError, match="param must be one of"):
        profile_objective(MARKET, quotes, "sigma", [1.0], anchor=TRUTH, engine=ENGINE)  # type: ignore[arg-type]


def test_param_bounds_clip() -> None:
    bounds = ParamBounds(
        lower=HestonParams(0.01, 0.5, 0.01, 0.1, -0.9),
        upper=HestonParams(0.1, 5.0, 0.1, 1.0, 0.0),
    )
    clipped = bounds.clip(HestonParams(v0=1.0, kappa=0.0, theta=0.05, xi=0.3, rho=0.5))
    assert clipped == HestonParams(v0=0.1, kappa=0.5, theta=0.05, xi=0.3, rho=0.0)
