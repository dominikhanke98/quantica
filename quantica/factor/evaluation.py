r"""Out-of-sample estimator comparison — the headline artifact of the factor step.

Estimating the covariance is the hard part; *which* estimator to trust is settled by
**out-of-sample risk forecasting**, not in-sample fit. This module is the
comparison-and-validation framework that neither scikit-learn nor the usual
portfolio libraries ship: fit each estimator on a training window, then measure how
well its covariance predicts the *realized* volatility of test portfolios in the
following, non-overlapping window.

The pieces:

* :func:`walk_forward_windows` — strictly non-overlapping train/test splits, with
  the no-lookahead property (``train_end == test_start``) made explicit and testable.
* :func:`compare_estimators` — the walk-forward loop. Per window it fits each
  estimator, then evaluates two families of test portfolios: **random** long-only
  portfolios (generic risk forecasting) and each estimator's own **minimum-variance**
  portfolio (the stress case, where estimation error bites hardest — Michaud's
  "error maximiser").
* The **bias statistic** ``realized / forecast`` volatility: a well-calibrated risk
  model has bias :math:`\approx 1`. Its whole *distribution* is reported, not just
  the mean — a model can be right on average and badly dispersed.
* :func:`min_variance_true_loss` / :func:`frobenius_error` — the known-truth losses:
  when the true covariance is known (synthetic data), each estimator's error can be
  measured *directly* against ground truth, not only through the OOS proxy.

The point is not to crown a single winner but to show **which estimator to trust
when**: shrinkage and the factor model earn their keep exactly where the sample
covariance fails — many assets, few observations, and portfolios that invert the
matrix.

References
----------
Michaud, R. (1989), "The Markowitz optimization enigma: is 'optimized' optimal?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from quantica.factor.estimators import min_variance_weights

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.factor.estimators import CovarianceEstimator

__all__ = [
    "BiasStats",
    "EstimatorComparison",
    "WalkForwardWindow",
    "compare_estimators",
    "frobenius_error",
    "min_variance_true_loss",
    "walk_forward_windows",
]

_DEFAULT_N_RANDOM = 25  # random test portfolios per window
_CALIBRATION_BAND = (0.8, 1.25)  # a bias in this range is "well calibrated"


class WalkForwardWindow(NamedTuple):
    """One train/test split. Invariant: ``train_end == test_start`` (no lookahead)."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_windows(
    n_obs: int, train_window: int, test_window: int
) -> tuple[WalkForwardWindow, ...]:
    """Rolling train windows, each immediately followed by its own test window.

    Test windows tile the timeline without overlap (each period is tested once);
    every window's training data lies strictly *before* its test data, so there is
    no lookahead. Returns as many windows as fit in ``n_obs``.
    """
    if train_window < 2 or test_window < 1:
        raise ValueError("need train_window >= 2 and test_window >= 1")
    if train_window + test_window > n_obs:
        raise ValueError(
            f"train_window + test_window = {train_window + test_window} exceeds n_obs = {n_obs}"
        )
    windows = []
    start = 0
    while start + train_window + test_window <= n_obs:
        train_end = start + train_window
        windows.append(
            WalkForwardWindow(
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test_window,
            )
        )
        start += test_window  # advance by the test length: test windows do not overlap
    return tuple(windows)


class BiasStats(NamedTuple):
    """Distribution of the ``realized / forecast`` volatility ratio for one estimator."""

    ratios: FloatArray

    @property
    def mean(self) -> float:
        return float(np.mean(self.ratios))

    @property
    def median(self) -> float:
        return float(np.median(self.ratios))

    @property
    def p05(self) -> float:
        return float(np.quantile(self.ratios, 0.05))

    @property
    def p95(self) -> float:
        return float(np.quantile(self.ratios, 0.95))

    @property
    def dispersion(self) -> float:
        """Interquantile spread (p95 - p05) — how *consistently* calibrated, not just on average."""
        return self.p95 - self.p05

    def fraction_calibrated(self, band: tuple[float, float] = _CALIBRATION_BAND) -> float:
        """Share of forecasts whose bias falls in a well-calibrated band."""
        lo, hi = band
        return float(np.mean((self.ratios >= lo) & (self.ratios <= hi)))


@dataclass(frozen=True)
class EstimatorComparison:
    """Walk-forward comparison of covariance estimators on OOS risk forecasting."""

    estimator_names: tuple[str, ...]
    windows: tuple[WalkForwardWindow, ...]
    #: Pooled random-portfolio bias ratios per estimator.
    bias: dict[str, BiasStats]
    #: Realized OOS volatility of each estimator's own min-variance portfolio, per window.
    min_variance_realized_vol: dict[str, FloatArray]
    #: Bias ratios of each estimator's min-variance portfolio, per window.
    min_variance_bias: dict[str, BiasStats]

    def mean_min_variance_vol(self) -> dict[str, float]:
        """Average realized OOS volatility of each estimator's min-variance portfolio."""
        return {name: float(np.mean(v)) for name, v in self.min_variance_realized_vol.items()}

    def best_min_variance_estimator(self) -> str:
        """The estimator whose min-variance portfolio realized the lowest OOS volatility."""
        means = self.mean_min_variance_vol()
        return min(means, key=lambda k: means[k])


def compare_estimators(
    asset_returns: FloatArray,
    estimators: tuple[CovarianceEstimator, ...],
    *,
    train_window: int,
    test_window: int,
    factor_returns: FloatArray | None = None,
    rng: np.random.Generator,
    n_random_portfolios: int = _DEFAULT_N_RANDOM,
) -> EstimatorComparison:
    r"""Walk-forward out-of-sample comparison of covariance estimators.

    For each non-overlapping window, every estimator is fitted on the training
    slice and scored on the *following* test slice two ways: the realized-vs-forecast
    volatility **bias** on a shared set of random long-only portfolios, and the
    realized volatility of the estimator's own **minimum-variance** portfolio.

    The random portfolios are drawn once per window (seeded) and shared across
    estimators, so the bias comparison is paired. No data from a test window ever
    enters an estimator's fit — the windows enforce it.
    """
    r = np.asarray(asset_returns, dtype=np.float64)
    f = None if factor_returns is None else np.asarray(factor_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("asset_returns must be 2-D (T, n)")
    if f is not None and f.shape[0] != r.shape[0]:
        raise ValueError("factor_returns must have the same number of rows as asset_returns")
    n_assets = r.shape[1]
    windows = walk_forward_windows(r.shape[0], train_window, test_window)

    bias_pool: dict[str, list[float]] = {e.name: [] for e in estimators}
    mv_vol: dict[str, list[float]] = {e.name: [] for e in estimators}
    mv_bias: dict[str, list[float]] = {e.name: [] for e in estimators}

    for window in windows:
        r_train = r[window.train_start : window.train_end]
        r_test = r[window.test_start : window.test_end]
        f_train = None if f is None else f[window.train_start : window.train_end]
        # Long-only random portfolios, shared across estimators (paired comparison).
        random_weights = rng.dirichlet(np.ones(n_assets), size=n_random_portfolios)

        for estimator in estimators:
            cov = estimator.estimate(r_train, f_train)
            for w in random_weights:
                bias_pool[estimator.name].append(_bias_ratio(cov, w, r_test))
            w_mv = min_variance_weights(cov)
            mv_vol[estimator.name].append(_realized_vol(w_mv, r_test))
            mv_bias[estimator.name].append(_bias_ratio(cov, w_mv, r_test))

    return EstimatorComparison(
        estimator_names=tuple(e.name for e in estimators),
        windows=windows,
        bias={n: BiasStats(np.array(v, dtype=np.float64)) for n, v in bias_pool.items()},
        min_variance_realized_vol={n: np.array(v, dtype=np.float64) for n, v in mv_vol.items()},
        min_variance_bias={n: BiasStats(np.array(v, dtype=np.float64)) for n, v in mv_bias.items()},
    )


# --------------------------------------------------------------------------- #
# Known-truth losses (synthetic data, where the true covariance is available)
# --------------------------------------------------------------------------- #


def frobenius_error(estimate: FloatArray, truth: FloatArray) -> float:
    r"""Frobenius norm :math:`\lVert \hat\Sigma - \Sigma \rVert_F` vs a known covariance."""
    return float(np.linalg.norm(np.asarray(estimate) - np.asarray(truth), ord="fro"))


def min_variance_true_loss(estimate: FloatArray, truth: FloatArray) -> float:
    r"""True variance of the min-variance portfolio built from ``estimate``.

    Build :math:`w = \arg\min w^\top \hat\Sigma w` (from the *estimated* covariance),
    then evaluate its variance under the *true* covariance, :math:`w^\top \Sigma w`.
    This is the portfolio-relevant loss: the estimator whose min-variance portfolio
    has the lowest *true* variance is the one you would have wanted — and on a known
    truth it is measured directly, with no OOS sampling noise.
    """
    w = min_variance_weights(estimate)
    return float(w @ np.asarray(truth, dtype=np.float64) @ w)


def _bias_ratio(cov: FloatArray, weights: FloatArray, test_returns: FloatArray) -> float:
    forecast = _forecast_vol(cov, weights)
    realized = _realized_vol(weights, test_returns)
    return realized / forecast if forecast > 0.0 else float("nan")


def _forecast_vol(cov: FloatArray, weights: FloatArray) -> float:
    return float(np.sqrt(max(float(weights @ cov @ weights), 0.0)))


def _realized_vol(weights: FloatArray, test_returns: FloatArray) -> float:
    portfolio_returns = np.asarray(test_returns, dtype=np.float64) @ weights
    return float(np.std(portfolio_returns, ddof=1))
