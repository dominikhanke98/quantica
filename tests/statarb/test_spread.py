"""Validation of the Ornstein--Uhlenbeck spread model (numerical-validation skill).

Known-truth recovery: on simulated OU paths with planted parameters, the AR(1)
estimator recovers the mean-reversion speed, long-run mean, volatility and half-life. The
half-life formula ``ln 2 / kappa`` is checked directly, and a non-mean-reverting series (a
random walk) is correctly reported with an infinite half-life rather than a spurious
finite one.
"""

from __future__ import annotations

import numpy as np
from quantica.statarb import (
    estimate_ou_process,
    generate_independent_random_walks,
    ou_half_life,
    simulate_ou_process,
)


def test_ou_recovers_planted_parameters() -> None:
    """Averaged over paths, the estimator recovers the planted OU parameters.

    A single AR(1) fit carries the small-sample (Kendall) bias in the autoregressive
    coefficient, so the recovery is checked on the mean over many seeded paths — where the
    estimates concentrate on the truth.
    """
    kappa, mu, sigma = 0.15, 3.0, 0.5
    kappas, mus, sigmas, half_lives = [], [], [], []
    for seed in range(40):
        path = simulate_ou_process(
            2000, np.random.default_rng(seed), kappa=kappa, mu=mu, sigma=sigma
        )
        fit = estimate_ou_process(path)
        kappas.append(fit.mean_reversion_speed)
        mus.append(fit.long_run_mean)
        sigmas.append(fit.volatility)
        half_lives.append(fit.half_life)
    assert abs(np.mean(kappas) - kappa) < 0.02  # speed
    assert abs(np.mean(mus) - mu) < 0.1  # long-run mean
    assert abs(np.mean(sigmas) - sigma) < 0.02  # volatility
    assert abs(np.mean(half_lives) - ou_half_life(kappa)) < 0.6  # half-life ~ 4.62


def test_half_life_formula() -> None:
    """The half-life is exactly ln(2) / kappa, and infinite for a non-reverting series."""
    assert np.isclose(ou_half_life(0.1), np.log(2.0) / 0.1)
    assert np.isclose(ou_half_life(0.5), np.log(2.0) / 0.5)
    assert ou_half_life(0.0) == float("inf")
    assert ou_half_life(-0.3) == float("inf")


def test_estimated_half_life_matches_speed() -> None:
    """The reported half-life is consistent with the reported mean-reversion speed."""
    path = simulate_ou_process(5000, np.random.default_rng(1), kappa=0.2, mu=0.0, sigma=1.0)
    fit = estimate_ou_process(path)
    assert np.isclose(fit.half_life, ou_half_life(fit.mean_reversion_speed))


def test_ar1_coefficient_matches_exp_minus_kappa() -> None:
    """The fitted AR(1) coefficient recovers phi = exp(-kappa*dt)."""
    kappa = 0.25
    fits = [
        estimate_ou_process(
            simulate_ou_process(3000, np.random.default_rng(s), kappa=kappa, mu=1.0, sigma=0.4)
        ).ar1_coefficient
        for s in range(20)
    ]
    assert abs(np.mean(fits) - np.exp(-kappa)) < 0.01


def test_random_walk_has_untradeable_half_life() -> None:
    """A random walk (phi ~ 1) yields a half-life vastly beyond any tradeable horizon.

    The AR(1) coefficient of a unit-root series estimates near 1, so the implied half-life
    is enormous (or infinite when phi rounds above 1) — orders of magnitude past the ~5-20
    period half-life of a genuinely mean-reverting spread, which is how the screen rejects
    non-reverting "pairs".
    """
    walk = generate_independent_random_walks(2000, 1, np.random.default_rng(2))[:, 0]
    fit = estimate_ou_process(walk)
    assert fit.half_life > 100.0  # untradeable vs a real spread's single-digit half-life
    assert fit.ar1_coefficient > 0.99  # essentially a unit root


def test_rejects_bad_inputs() -> None:
    """Too-short series and non-positive dt are rejected."""
    import pytest

    with pytest.raises(ValueError, match="length >= 3"):
        estimate_ou_process(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="dt must be positive"):
        estimate_ou_process(np.zeros(10), dt=0.0)
