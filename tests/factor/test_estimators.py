r"""The three covariance estimators — anchored to their libraries, plus GMV algebra.

Scope discipline (stage 2): we do not re-implement the estimators, so the tests
confirm we *wrap* them correctly — sample == ``numpy.cov``, Ledoit--Wolf ==
``sklearn`` directly, factor == the stage-1 model covariance — and that the
minimum-variance and conditioning helpers match their closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.factor import (
    FactorCovariance,
    FactorRiskModel,
    LedoitWolfCovariance,
    SampleCovariance,
    condition_number,
    generate_factor_data,
    min_variance_weights,
)

sklearn_cov = pytest.importorskip("sklearn.covariance")


def _panel(n_assets: int = 8, n_periods: int = 200, seed: int = 0):  # type: ignore[no-untyped-def]
    betas = np.random.default_rng(seed).normal(1.0, 0.4, size=(n_assets, 4))
    return generate_factor_data(n_periods, betas, np.random.default_rng(seed + 1))


# --------------------------------------------------------------------------- #
# Estimators anchored to their libraries
# --------------------------------------------------------------------------- #


def test_sample_covariance_matches_numpy() -> None:
    data = _panel()
    got = SampleCovariance().estimate(data.asset_returns)
    np.testing.assert_allclose(got, np.cov(data.asset_returns, rowvar=False), atol=1e-14)


def test_ledoit_wolf_matches_sklearn() -> None:
    data = _panel()
    got = LedoitWolfCovariance().estimate(data.asset_returns)
    expected = sklearn_cov.LedoitWolf().fit(data.asset_returns).covariance_
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_factor_covariance_matches_stage1_model() -> None:
    data = _panel()
    got = FactorCovariance(factor_names=data.factor_names).estimate(
        data.asset_returns, data.factor_returns
    )
    expected = FactorRiskModel.fit(
        data.asset_returns, data.factor_returns, factor_names=data.factor_names
    ).covariance()
    np.testing.assert_allclose(got, expected, atol=1e-15)


def test_factor_covariance_requires_factors() -> None:
    data = _panel()
    with pytest.raises(ValueError, match="requires factor_returns"):
        FactorCovariance().estimate(data.asset_returns, None)


def test_estimators_expose_names() -> None:
    assert SampleCovariance().name == "sample"
    assert LedoitWolfCovariance().name == "ledoit-wolf"
    assert FactorCovariance().name == "factor"


# --------------------------------------------------------------------------- #
# Minimum-variance weights and condition number
# --------------------------------------------------------------------------- #


def test_min_variance_weights_diagonal_closed_form() -> None:
    # For a diagonal covariance, GMV weights are proportional to inverse variance.
    cov = np.diag([1.0, 4.0, 25.0])
    w = min_variance_weights(cov)
    expected = np.array([1.0, 0.25, 0.04])
    expected /= expected.sum()
    np.testing.assert_allclose(w, expected, atol=1e-14)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_weights_match_formula() -> None:
    data = _panel()
    cov = SampleCovariance().estimate(data.asset_returns)
    ones = np.ones(cov.shape[0])
    z = np.linalg.solve(cov, ones)
    np.testing.assert_allclose(min_variance_weights(cov), z / z.sum(), atol=1e-12)


def test_min_variance_weights_minimize_variance() -> None:
    # The GMV portfolio has no lower-variance neighbour: perturbing it (keeping the
    # budget) can only raise w' Sigma w.
    data = _panel()
    cov = SampleCovariance().estimate(data.asset_returns)
    w = min_variance_weights(cov)
    base = float(w @ cov @ w)
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = rng.normal(size=w.size)
        d -= d.mean()  # keep the weights summing to one
        assert float((w + 1e-3 * d) @ cov @ (w + 1e-3 * d)) >= base - 1e-15


def test_condition_number_of_diagonal() -> None:
    assert condition_number(np.diag([1.0, 100.0])) == pytest.approx(100.0)
    assert condition_number(np.eye(5)) == pytest.approx(1.0)
