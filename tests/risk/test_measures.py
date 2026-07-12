"""VaR/ES measure definitions — closed forms and the empirical estimator."""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk import empirical_var_es, normal_var_es
from scipy.stats import norm


@pytest.mark.parametrize("level", [0.95, 0.975, 0.99])
def test_normal_var_es_matches_closed_form(level: float) -> None:
    # Standard normal P&L: VaR = z_alpha, ES = phi(z_alpha)/(1-alpha).
    est = normal_var_es(0.0, 1.0, level)
    z = norm.ppf(level)
    assert est.var == pytest.approx(z)
    assert est.es == pytest.approx(norm.pdf(z) / (1.0 - level))
    assert est.es > est.var  # ES is always at least VaR


def test_normal_var_es_shifts_with_mean_and_scales_with_sigma() -> None:
    base = normal_var_es(0.0, 1.0, 0.99)
    # A positive drift lowers loss-based VaR one-for-one; sigma scales the tail.
    assert normal_var_es(0.5, 1.0, 0.99).var == pytest.approx(base.var - 0.5)
    assert normal_var_es(0.0, 3.0, 0.99).var == pytest.approx(3.0 * base.var)


def test_var_and_es_increase_with_confidence() -> None:
    v95 = normal_var_es(0.0, 1.0, 0.95)
    v99 = normal_var_es(0.0, 1.0, 0.99)
    assert v99.var > v95.var and v99.es > v95.es


def test_empirical_converges_to_normal_closed_form() -> None:
    rng = np.random.default_rng(0)
    losses = rng.normal(0.0, 1.0, 500_000)
    est = empirical_var_es(losses, 0.99)
    ref = normal_var_es(0.0, 1.0, 0.99)
    assert est.var == pytest.approx(ref.var, rel=0.03)
    assert est.es == pytest.approx(ref.es, rel=0.03)


def test_empirical_es_at_least_var() -> None:
    rng = np.random.default_rng(1)
    losses = rng.standard_t(4, 10_000)
    est = empirical_var_es(losses, 0.975)
    assert est.es >= est.var


@pytest.mark.parametrize("level", [-0.1, 0.0, 1.0, 1.5])
def test_rejects_bad_level(level: float) -> None:
    with pytest.raises(ValueError, match="level must be in"):
        normal_var_es(0.0, 1.0, level)
    with pytest.raises(ValueError, match="level must be in"):
        empirical_var_es(np.zeros(10), level)


def test_empirical_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        empirical_var_es(np.array([]), 0.99)
