"""Validation of the statistical (PCA) factor model (numerical-validation skill).

The checks mirror the "recover the truth, anchor the algebra, tie back to the observable
model" structure the factor step uses throughout:

* **Known-truth recovery (headline)** — on returns generated from a known low-rank factor
  structure with a clean spectral gap, the Marchenko--Pastur cutoff recovers the true
  *number* of factors and the recovered loadings span the true factor *subspace* (checked
  by principal angles); pure noise yields zero statistical factors.
* **Anchors** — the reconstructed covariance is symmetric PSD, the variance-explained
  ratios sum to one, the diagonal reproduces the sample variances exactly, a full-rank fit
  reproduces the sample covariance, and the eigendecomposition matches an independent SVD.
* **Selection rules** — variance-explained monotonicity, the scree elbow on a two-block
  spectrum, and the Marchenko--Pastur edge formula; the bulk-variance refit recovers weak
  factors the plain cutoff misses under a dominant market mode.
* **Tie-back** — the PCA covariance plugs into the stage-2 out-of-sample comparison and,
  on a factor-structured universe, forecasts risk as well as the observable factor model
  and far better than the sample covariance.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.factor import (
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    StatisticalFactorCovariance,
    StatisticalFactorModel,
    compare_estimators,
    generate_factor_data,
    marchenko_pastur_edges,
    marchenko_pastur_rank,
    min_variance_true_loss,
    scree_elbow_rank,
    subspace_similarity,
    variance_explained_rank,
)


def _spectral_gap_panel(seed: int, n: int = 40, t: int = 600, k: int = 3):  # type: ignore[no-untyped-def]
    """Mixed-sign, equal-strength factors + low noise -> a clean gap above the MP edge."""
    rng = np.random.default_rng(seed)
    betas = rng.uniform(-1.0, 1.0, size=(n, k))
    return generate_factor_data(
        t,
        betas,
        rng,
        specific_vols=0.04,
        factor_vols=np.array([0.09] * k),
        factor_names=tuple(f"F{i}" for i in range(k)),
    )


def _dominant_market_panel(seed: int, n: int = 40, t: int = 500, k: int = 3):  # type: ignore[no-untyped-def]
    """All-positive loadings -> one dominant "market" mode that depletes the bulk."""
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.3, 1.3, size=(n, k))
    return generate_factor_data(
        t,
        betas,
        rng,
        specific_vols=0.05,
        factor_vols=np.array([0.05] * k),
        factor_names=tuple(f"F{i}" for i in range(k)),
    )


# --------------------------------------------------------------------------- #
# Known-truth recovery (the headline)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_marchenko_pastur_recovers_true_factor_count_and_subspace(seed: int) -> None:
    """PCA recovers the true factor count (MP cutoff) and the true loading subspace."""
    data = _spectral_gap_panel(seed)
    model = StatisticalFactorModel.fit(data.asset_returns)  # default: MP, sigma^2 = 1
    assert model.n_factors == data.true_betas.shape[1]  # exactly the planted count
    assert model.selection == "marchenko_pastur"
    # The recovered loadings span the planted factor space (individual PCs are not identified).
    assert subspace_similarity(model.betas, data.true_betas) > 0.99


def test_pure_noise_has_no_statistical_factors() -> None:
    """On i.i.d. noise the Marchenko--Pastur cutoff certifies zero real factors."""
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((400, 40))
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(noise, rowvar=False))
    assert marchenko_pastur_rank(eigenvalues, 40, 400) == 0


# --------------------------------------------------------------------------- #
# Reconstruction anchors
# --------------------------------------------------------------------------- #


def test_reconstructed_covariance_is_symmetric_psd() -> None:
    """Sigma = B Bᵀ + D is symmetric and positive semidefinite by construction."""
    model = StatisticalFactorModel.fit(_spectral_gap_panel(0).asset_returns)
    sigma = model.covariance()
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > -1e-10


def test_variance_explained_ratio_sums_to_one() -> None:
    """The full eigenvalue spectrum partitions the total variance."""
    model = StatisticalFactorModel.fit(_spectral_gap_panel(1).asset_returns)
    assert np.isclose(model.explained_variance_ratio.sum(), 1.0, atol=1e-12)
    assert np.isclose(model.cumulative_variance_ratio[-1], 1.0, atol=1e-12)


def test_diagonal_preserves_sample_variances_exactly() -> None:
    """Correlation-PCA folds communality + specific to one, so diag(Sigma) == sample var."""
    data = _spectral_gap_panel(2)
    model = StatisticalFactorModel.fit(data.asset_returns, n_components=3)
    sample_var = np.var(data.asset_returns, axis=0, ddof=1)
    assert np.allclose(np.diag(model.covariance()), sample_var, atol=1e-12)


def test_full_rank_reproduces_sample_covariance() -> None:
    """With k = n the reconstruction is exact — it *is* the sample covariance."""
    data = _spectral_gap_panel(3, n=25, t=300)
    model = StatisticalFactorModel.fit(data.asset_returns, n_components=25)
    assert np.allclose(model.covariance(), np.cov(data.asset_returns, rowvar=False), atol=1e-10)


def test_pca_matches_independent_svd() -> None:
    """Eigendecomposition of the correlation matrix == SVD of the standardized returns."""
    data = _spectral_gap_panel(0)
    r = data.asset_returns
    standardized = (r - r.mean(axis=0)) / r.std(axis=0, ddof=1)
    singular_values = np.linalg.svd(standardized / np.sqrt(len(r) - 1), compute_uv=False)
    model = StatisticalFactorModel.fit(r, n_components=1)
    top = np.sort(model.eigenvalues)[::-1][:10]
    assert np.allclose(top, np.sort(singular_values**2)[::-1][:10], atol=1e-9)


def test_single_factor_case() -> None:
    """A one-factor fit yields rank-1 loadings and a scalar factor covariance."""
    data = _spectral_gap_panel(0, k=1)
    model = StatisticalFactorModel.fit(data.asset_returns, n_components=1)
    assert model.betas.shape[1] == 1
    assert model.factor_cov.shape == (1, 1)
    assert np.isclose(model.factor_cov[0, 0], 1.0)
    assert subspace_similarity(model.betas, data.true_betas) > 0.99


# --------------------------------------------------------------------------- #
# Component-selection rules
# --------------------------------------------------------------------------- #


def test_variance_explained_rank_monotone_in_threshold() -> None:
    """A higher variance target never asks for fewer components; 1.0 asks for all."""
    eigenvalues = np.array([5.0, 3.0, 1.5, 0.3, 0.2])
    assert variance_explained_rank(eigenvalues, 0.5) <= variance_explained_rank(eigenvalues, 0.95)
    assert variance_explained_rank(eigenvalues, 1.0) == eigenvalues.size


def test_scree_elbow_on_two_block_spectrum() -> None:
    """A spectrum of three large then many tiny eigenvalues elbows at three."""
    eigenvalues = np.array([20.0, 15.0, 10.0, 0.4, 0.35, 0.3, 0.25, 0.2])
    assert scree_elbow_rank(eigenvalues) == 3


def test_marchenko_pastur_edges_formula() -> None:
    """The edges match sigma^2 (1 +/- sqrt(n/T))^2."""
    lo, hi = marchenko_pastur_edges(40, 400, variance=2.0)
    q = np.sqrt(40 / 400)
    assert np.isclose(lo, 2.0 * (1 - q) ** 2)
    assert np.isclose(hi, 2.0 * (1 + q) ** 2)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_variance_adjustment_recovers_factors_the_plain_cutoff_misses(seed: int) -> None:
    """Under a dominant market mode sigma^2=1 under-counts; the bulk refit recovers the rest.

    An all-positive-loading universe has one huge "market" eigenvalue that steals variance
    from the bulk, so the plain sigma^2=1 edge sits too high and the weaker real factors hide
    inside it. Re-estimating the bulk variance lowers the edge and recovers them.
    """
    data = _dominant_market_panel(seed)
    k_true = data.true_betas.shape[1]
    eigenvalues = np.linalg.eigvalsh(np.corrcoef(data.asset_returns, rowvar=False))
    plain = marchenko_pastur_rank(eigenvalues, 40, 500, adjust_variance=False)
    adjusted = marchenko_pastur_rank(eigenvalues, 40, 500, adjust_variance=True)
    assert plain < k_true <= adjusted  # plain misses weak factors; the refit brackets the truth


# --------------------------------------------------------------------------- #
# Tie-back: the PCA covariance in the stage-2 out-of-sample comparison
# --------------------------------------------------------------------------- #


def test_statistical_factor_covariance_estimator_interface() -> None:
    """The estimator returns a symmetric PSD (n, n) covariance and needs no factor returns."""
    data = _spectral_gap_panel(0)
    cov = StatisticalFactorCovariance(n_components=3).estimate(data.asset_returns)
    assert cov.shape == (data.asset_returns.shape[1],) * 2
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_statistical_factor_beats_sample_on_known_truth() -> None:
    """On a factor-structured universe the PCA min-variance portfolio has lower *true* loss."""
    rng = np.random.default_rng(0)
    betas = rng.uniform(0.3, 1.2, size=(40, 4))
    data = generate_factor_data(400, betas, rng, specific_vols=0.05)
    true_cov = betas @ np.diag(np.array([0.045, 0.03, 0.03, 0.045]) ** 2) @ betas.T + np.diag(
        np.full(40, 0.05**2)
    )
    train = data.asset_returns[:60]
    sample_loss = min_variance_true_loss(SampleCovariance().estimate(train), true_cov)
    pca_loss = min_variance_true_loss(
        StatisticalFactorCovariance(n_components=4).estimate(train), true_cov
    )
    assert pca_loss < sample_loss


def test_statistical_factor_competes_in_out_of_sample_comparison() -> None:
    """Plugged into compare_estimators, the PCA model beats sample OOS when assets crowd T."""
    rng = np.random.default_rng(0)
    betas = rng.uniform(0.3, 1.2, size=(40, 4))
    data = generate_factor_data(400, betas, rng, specific_vols=0.05)
    comparison = compare_estimators(
        data.asset_returns,
        (
            SampleCovariance(),
            LedoitWolfCovariance(),
            StatisticalFactorCovariance(n_components=4),
            FactorCovariance(),
        ),
        train_window=55,
        test_window=12,
        factor_returns=data.factor_returns,
        rng=np.random.default_rng(1),
    )
    vols = comparison.mean_min_variance_vol()
    assert vols["statistical-factor"] < vols["sample"]
    # The statistical model tracks the observable factor model (same DGP) far below sample.
    assert vols["statistical-factor"] < vols["ledoit-wolf"]


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_rejects_bad_inputs() -> None:
    """1-D returns, too-few observations, and out-of-range component counts are rejected."""
    rng = np.random.default_rng(0)
    good = rng.standard_normal((100, 5))
    with pytest.raises(ValueError, match="2-D"):
        StatisticalFactorModel.fit(rng.standard_normal(100))
    with pytest.raises(ValueError, match="at least 2"):
        StatisticalFactorModel.fit(rng.standard_normal((1, 5)))
    with pytest.raises(ValueError, match="n_components"):
        StatisticalFactorModel.fit(good, n_components=6)
    with pytest.raises(ValueError, match="threshold"):
        variance_explained_rank(np.array([1.0, 0.5]), threshold=1.5)
