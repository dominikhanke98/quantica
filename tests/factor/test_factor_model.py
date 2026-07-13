r"""Validation of the factor risk model (numerical-validation skill, stage 1).

- **Betas anchored to an independent OLS**: the statsmodels-fitted loadings equal
  a direct ``numpy.linalg.lstsq`` solve to machine precision — "we called
  statsmodels correctly" is itself checked, not assumed.
- **Single-factor reduces to the CAPM beta** ``cov(r, mkt) / var(mkt)``.
- **Known-truth recovery (the headline)**: planted betas and specific variances
  are recovered from a synthetic panel within their standard errors.
- **Structural**: :math:`\Sigma = B F B^\top + D` is symmetric PSD and equals its
  definition; the variance and portfolio-risk decompositions add up.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.factor import (
    FactorRiskModel,
    estimate_exposures,
    generate_factor_data,
)

statsmodels_api = pytest.importorskip("statsmodels.api")

_BETAS = np.array(
    [
        [1.10, 0.50, -0.30, 0.20],
        [0.80, -0.20, 0.60, 0.00],
        [1.30, 0.90, 0.10, -0.40],
        [0.50, 0.00, 0.00, 0.70],
        [1.00, 0.30, 0.30, 0.30],
    ]
)
_FACTORS = ("MKT", "SMB", "HML", "MOM")


def make_model(n_periods: int = 3000, specific_vol: float = 0.03, seed: int = 0):  # type: ignore[no-untyped-def]
    data = generate_factor_data(
        n_periods, _BETAS, np.random.default_rng(seed), specific_vols=specific_vol
    )
    model = FactorRiskModel.fit(
        data.asset_returns,
        data.factor_returns,
        asset_names=data.asset_names,
        factor_names=data.factor_names,
    )
    return model, data


# --------------------------------------------------------------------------- #
# 1. Estimator anchored to an independent OLS
# --------------------------------------------------------------------------- #


def test_betas_match_independent_lstsq() -> None:
    model, data = make_model()
    design = np.column_stack([np.ones(data.asset_returns.shape[0]), data.factor_returns])
    coef, *_ = np.linalg.lstsq(design, data.asset_returns, rcond=None)  # (k+1, n_assets)
    np.testing.assert_allclose(model.betas, coef[1:].T, atol=1e-10)
    np.testing.assert_allclose(model.alphas, coef[0], atol=1e-10)


def test_exposures_match_statsmodels_directly() -> None:
    _, data = make_model()
    y = data.asset_returns[:, 2]
    res = statsmodels_api.OLS(y, statsmodels_api.add_constant(data.factor_returns)).fit()
    ours = estimate_exposures(y, data.factor_returns, _FACTORS)
    np.testing.assert_allclose(ours.betas, np.asarray(res.params)[1:], atol=1e-12)
    assert ours.r_squared == pytest.approx(float(res.rsquared))
    assert ours.specific_variance == pytest.approx(float(res.mse_resid))


def test_single_factor_reduces_to_capm_beta() -> None:
    _, data = make_model()
    market = data.factor_returns[:, :1]
    for i in range(data.asset_returns.shape[1]):
        y = data.asset_returns[:, i]
        beta = estimate_exposures(y, market, ("MKT",)).betas[0]
        capm = np.cov(y, market[:, 0], ddof=1)[0, 1] / np.var(market[:, 0], ddof=1)
        assert beta == pytest.approx(float(capm), rel=1e-10)


def test_single_factor_model_fits() -> None:
    _, data = make_model()
    model = FactorRiskModel.fit(data.asset_returns, data.factor_returns[:, :1])
    assert model.betas.shape == (5, 1)
    assert model.factor_cov.shape == (1, 1)
    assert model.covariance().shape == (5, 5)


# --------------------------------------------------------------------------- #
# 2. Known-truth recovery
# --------------------------------------------------------------------------- #


def test_recovers_planted_betas_within_standard_errors() -> None:
    n_periods = 4000
    model, data = make_model(n_periods=n_periods, specific_vol=0.03, seed=1)
    design = np.column_stack([np.ones(n_periods), data.factor_returns])
    xtx_inv = np.linalg.inv(design.T @ design)
    for i in range(5):
        se = np.sqrt(model.specific_var[i] * np.diag(xtx_inv)[1:])  # per-beta std errors
        z = np.abs(model.betas[i] - _BETAS[i]) / se
        assert np.all(z < 4.0), f"asset {i}: standardised errors {z}"


def test_recovers_specific_variance() -> None:
    model, data = make_model(n_periods=4000, specific_vol=0.04, seed=2)
    np.testing.assert_allclose(model.specific_var, data.true_specific_var, rtol=0.1)


def test_t_stats_separate_real_from_absent_factors() -> None:
    # Asset loads on MKT only; the other three betas are zero by construction.
    betas = np.array([[1.0, 0.0, 0.0, 0.0]])
    data = generate_factor_data(4000, betas, np.random.default_rng(3), specific_vols=0.02)
    exp = estimate_exposures(data.asset_returns[:, 0], data.factor_returns, _FACTORS)
    assert abs(exp.t_stats[0]) > 20.0  # MKT overwhelmingly significant
    assert np.all(np.abs(exp.t_stats[1:]) < 4.0)  # the planted-zero factors are not


# --------------------------------------------------------------------------- #
# 3. Structural: Sigma = B F B^T + D, symmetric PSD; decompositions add up
# --------------------------------------------------------------------------- #


def test_covariance_is_symmetric_and_psd() -> None:
    model, _ = make_model()
    sigma = model.covariance()
    np.testing.assert_allclose(sigma, sigma.T, atol=0.0)  # symmetrised exactly
    assert np.min(np.linalg.eigvalsh(sigma)) > 0.0  # PD (D > 0)


def test_covariance_equals_its_definition() -> None:
    model, _ = make_model()
    expected = model.betas @ model.factor_cov @ model.betas.T + np.diag(model.specific_var)
    np.testing.assert_allclose(model.covariance(), expected, atol=1e-15)
    np.testing.assert_allclose(
        model.systematic_covariance(), model.betas @ model.factor_cov @ model.betas.T, atol=1e-15
    )
    # Factor covariance is exactly numpy's sample covariance of the factors.
    _, data = make_model()
    np.testing.assert_allclose(
        model.factor_cov, np.cov(data.factor_returns, rowvar=False, ddof=1), atol=1e-12
    )


def test_variance_decomposition_adds_up() -> None:
    model, _ = make_model()
    sigma_diag = np.diag(model.covariance())
    for i, dec in enumerate(model.variance_decomposition()):
        assert dec.systematic_variance + dec.specific_variance == pytest.approx(dec.total_variance)
        assert dec.total_variance == pytest.approx(float(sigma_diag[i]))
        assert 0.0 < dec.systematic_fraction < 1.0


def test_portfolio_variance_matches_quadratic_form() -> None:
    model, _ = make_model()
    w = np.array([0.3, 0.1, 0.2, 0.25, 0.15])
    assert model.portfolio_variance(w) == pytest.approx(float(w @ model.covariance() @ w))


def test_portfolio_risk_decomposition_adds_up() -> None:
    model, _ = make_model()
    w = np.array([0.4, -0.1, 0.3, 0.2, 0.2])
    dec = model.portfolio_risk_decomposition(w)
    assert dec.systematic_variance + dec.specific_variance == pytest.approx(dec.total_variance)
    assert dec.total_variance == pytest.approx(model.portfolio_variance(w))
    np.testing.assert_allclose(dec.factor_exposure, model.betas.T @ w, atol=1e-12)
    assert dec.total_volatility == pytest.approx(np.sqrt(dec.total_variance))


# --------------------------------------------------------------------------- #
# 4. Determinism and input validation
# --------------------------------------------------------------------------- #


def test_synthetic_data_is_seeded() -> None:
    a = generate_factor_data(200, _BETAS, np.random.default_rng(9))
    b = generate_factor_data(200, _BETAS, np.random.default_rng(9))
    np.testing.assert_array_equal(a.asset_returns, b.asset_returns)
    np.testing.assert_array_equal(a.factor_returns, b.factor_returns)


def test_generate_factor_data_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_periods"):
        generate_factor_data(0, _BETAS, rng)
    with pytest.raises(ValueError, match="betas must be 2-D"):
        generate_factor_data(100, np.ones(4), rng)
    with pytest.raises(ValueError, match="names"):
        generate_factor_data(100, np.ones((3, 2)), rng, factor_names=_FACTORS)


def test_estimate_exposures_validation() -> None:
    f = np.random.default_rng(0).normal(size=(100, 4))
    with pytest.raises(ValueError, match="1-D of length"):
        estimate_exposures(np.ones(50), f, _FACTORS)
    with pytest.raises(ValueError, match="names"):
        estimate_exposures(np.ones(100), f, ("MKT",))
    with pytest.raises(ValueError, match="more than"):
        estimate_exposures(np.ones(5), np.ones((5, 4)), _FACTORS)


def test_fit_validation() -> None:
    r = np.random.default_rng(0).normal(size=(200, 3))
    f = np.random.default_rng(1).normal(size=(200, 4))
    with pytest.raises(ValueError, match="time dimension mismatch"):
        FactorRiskModel.fit(r, f[:100])
    with pytest.raises(ValueError, match="factor_names"):
        FactorRiskModel.fit(r, f, factor_names=("MKT",))
    with pytest.raises(ValueError, match="asset_names"):
        FactorRiskModel.fit(r, f, asset_names=("only_one",))
    model = FactorRiskModel.fit(r, f)
    with pytest.raises(ValueError, match="weights must have shape"):
        model.portfolio_variance(np.ones(5))
