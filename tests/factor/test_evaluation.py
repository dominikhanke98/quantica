r"""The OOS estimator-comparison framework — validated (numerical-validation skill).

- **No lookahead**, made testable: every walk-forward window trains strictly before
  it tests, and the test windows never overlap.
- **Known-truth losses**: on a synthetic factor DGP the *true* covariance is known,
  so each estimator's min-variance loss is measured directly against ground truth —
  the sample covariance is worst and the (correctly specified) factor model best.
- **Ill-conditioning**: the sample covariance's condition number blows up as the
  asset count approaches the observation count, while shrinkage and the factor model
  stay bounded — the concrete mechanism behind why sample covariance fails.
- **The min-variance stress (the headline)**: out of sample, the sample covariance's
  own min-variance portfolio has the worst realized volatility and a wildly
  optimistic forecast (bias ≫ 1) — Michaud's error maximiser — while on *random*
  portfolios the estimators are indistinguishable. That contrast is the finding:
  which estimator to trust depends on whether you invert the matrix.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from quantica.factor import (
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    compare_estimators,
    condition_number,
    frobenius_error,
    generate_factor_data,
    min_variance_true_loss,
    walk_forward_windows,
)

pytest.importorskip("sklearn.covariance")

_ESTIMATORS = (SampleCovariance(), LedoitWolfCovariance(), FactorCovariance())


def _true_factor_covariance(betas: np.ndarray, factor_vols: np.ndarray, specific_vol: float):  # type: ignore[no-untyped-def]
    return betas @ np.diag(factor_vols**2) @ betas.T + np.diag(
        np.full(betas.shape[0], specific_vol**2)
    )


# --------------------------------------------------------------------------- #
# 1. No lookahead
# --------------------------------------------------------------------------- #


def test_walk_forward_windows_have_no_lookahead() -> None:
    windows = walk_forward_windows(n_obs=200, train_window=48, test_window=12)
    assert len(windows) > 1
    for w in windows:
        assert w.train_start < w.train_end == w.test_start < w.test_end  # train before test
        assert w.train_end - w.train_start == 48
        assert w.test_end - w.test_start == 12
    # Test windows tile the timeline without overlap.
    for a, b in pairwise(windows):
        assert b.test_start == a.test_end


def test_walk_forward_windows_validation() -> None:
    with pytest.raises(ValueError, match="train_window >= 2"):
        walk_forward_windows(100, 1, 10)
    with pytest.raises(ValueError, match="exceeds n_obs"):
        walk_forward_windows(50, 40, 20)


# --------------------------------------------------------------------------- #
# 2. Known-truth losses
# --------------------------------------------------------------------------- #


def test_frobenius_and_min_variance_loss_hand_examples() -> None:
    a = np.diag([1.0, 1.0])
    b = np.diag([1.0, 2.0])
    assert frobenius_error(a, b) == pytest.approx(1.0)
    # min-var from diag([1,1]) -> w=[.5,.5]; true variance under diag([4,4]) = 2.0.
    assert min_variance_true_loss(np.diag([1.0, 1.0]), np.diag([4.0, 4.0])) == pytest.approx(2.0)


def test_known_truth_min_variance_ordering() -> None:
    # Synthetic factor DGP (true covariance known), few observations. The sample
    # covariance's min-variance portfolio has the highest TRUE variance; the
    # correctly specified factor model the lowest.
    factor_vols = np.array([0.05, 0.03, 0.03, 0.04])
    betas = np.random.default_rng(7).normal(1.0, 0.4, size=(30, 4))
    true_sigma = _true_factor_covariance(betas, factor_vols, 0.04)
    data = generate_factor_data(
        50, betas, np.random.default_rng(8), factor_vols=factor_vols, specific_vols=0.04
    )
    loss = {
        e.name: min_variance_true_loss(
            e.estimate(data.asset_returns, data.factor_returns), true_sigma
        )
        for e in _ESTIMATORS
    }
    assert loss["factor"] < loss["ledoit-wolf"] < loss["sample"]


# --------------------------------------------------------------------------- #
# 3. Ill-conditioning
# --------------------------------------------------------------------------- #


def test_sample_covariance_ill_conditions_as_assets_approach_observations() -> None:
    rng = np.random.default_rng(9)
    betas_all = rng.normal(1.0, 0.4, size=(55, 4))

    def cond(n_assets: int, name: str) -> float:
        data = generate_factor_data(60, betas_all[:n_assets], np.random.default_rng(n_assets))
        est = {"sample": SampleCovariance(), "lw": LedoitWolfCovariance(), "f": FactorCovariance()}[
            name
        ]
        return condition_number(est.estimate(data.asset_returns, data.factor_returns))

    # Sample condition number grows sharply with the asset count ...
    assert cond(50, "sample") > 10.0 * cond(10, "sample")
    # ... and near n = T it dwarfs shrinkage and the factor model, which stay bounded.
    assert cond(50, "sample") > 5.0 * cond(50, "lw")
    assert cond(50, "sample") > 5.0 * cond(50, "f")
    assert cond(50, "f") < cond(50, "lw")  # factor best-conditioned (low-rank + diagonal)


# --------------------------------------------------------------------------- #
# 4. The min-variance stress (the headline) vs random portfolios
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def comparison():  # type: ignore[no-untyped-def]
    # n = 30 assets, train window 45 (n/T = 0.67) — the regime where estimation
    # error bites. Deterministic.
    data = generate_factor_data(
        240, np.random.default_rng(20).normal(1.0, 0.4, size=(30, 4)), np.random.default_rng(21)
    )
    return compare_estimators(
        data.asset_returns,
        _ESTIMATORS,
        train_window=45,
        test_window=12,
        factor_returns=data.factor_returns,
        rng=np.random.default_rng(22),
        n_random_portfolios=15,
    )


def test_sample_min_variance_portfolio_is_worst_out_of_sample(comparison) -> None:  # type: ignore[no-untyped-def]
    mv = comparison.mean_min_variance_vol()
    assert mv["sample"] > mv["factor"]
    assert mv["sample"] > mv["ledoit-wolf"]
    assert comparison.best_min_variance_estimator() != "sample"


def test_sample_min_variance_forecast_is_optimistically_biased(comparison) -> None:  # type: ignore[no-untyped-def]
    # The min-variance optimiser minimises IN-sample variance, so the forecast is
    # too low and realized/forecast bias >> 1 — worst for the sample covariance.
    assert comparison.min_variance_bias["sample"].mean > 2.0
    assert comparison.min_variance_bias["sample"].mean > comparison.min_variance_bias["factor"].mean


def test_random_portfolios_are_indistinguishable(comparison) -> None:
    # The honest other half of the finding: on generic (non-optimised) portfolios
    # every estimator is roughly unbiased and they barely differ. The difference is
    # created by inverting the matrix, not by the matrix itself.
    for name in comparison.estimator_names:
        assert 0.9 < comparison.bias[name].mean < 1.1
    means = [comparison.bias[n].mean for n in comparison.estimator_names]
    assert max(means) - min(means) < 0.1


def test_bias_stats_distribution_fields(comparison) -> None:
    stats = comparison.bias["sample"]
    assert stats.p05 < stats.median < stats.p95
    assert 0.0 <= stats.fraction_calibrated() <= 1.0
    assert stats.dispersion == pytest.approx(stats.p95 - stats.p05)


# --------------------------------------------------------------------------- #
# 5. Determinism and validation
# --------------------------------------------------------------------------- #


def test_compare_estimators_is_deterministic() -> None:
    data = generate_factor_data(
        120, np.random.default_rng(0).normal(1.0, 0.4, size=(10, 4)), np.random.default_rng(1)
    )
    kwargs = dict(  # noqa: C408
        train_window=40, test_window=12, factor_returns=data.factor_returns, n_random_portfolios=10
    )
    a = compare_estimators(data.asset_returns, _ESTIMATORS, rng=np.random.default_rng(5), **kwargs)
    b = compare_estimators(data.asset_returns, _ESTIMATORS, rng=np.random.default_rng(5), **kwargs)
    for name in a.estimator_names:
        np.testing.assert_array_equal(a.bias[name].ratios, b.bias[name].ratios)
        np.testing.assert_array_equal(
            a.min_variance_realized_vol[name], b.min_variance_realized_vol[name]
        )


def test_compare_estimators_validation() -> None:
    data = generate_factor_data(
        120, np.random.default_rng(0).normal(1.0, 0.4, size=(6, 4)), np.random.default_rng(1)
    )
    with pytest.raises(ValueError, match="2-D"):
        compare_estimators(
            data.asset_returns[:, 0],
            _ESTIMATORS,
            train_window=40,
            test_window=12,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="same number of rows"):
        compare_estimators(
            data.asset_returns,
            _ESTIMATORS,
            train_window=40,
            test_window=12,
            factor_returns=data.factor_returns[:50],
            rng=np.random.default_rng(0),
        )
