"""Compute for the capital-markets view — covariance study, Jagannathan--Ma, DSR/PBO.

Pure orchestration of ``quantica.factor`` and ``quantica.portfolio`` on the bundled
Fama--French sample: it runs the out-of-sample estimator comparison, the no-short-sale
regularization result, and the overfitting search, returning plot-ready data. No
estimator or statistic is implemented here (CLAUDE.md §2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quantica.factor import (
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    compare_estimators,
    condition_number,
    min_variance_weights,
    walk_forward_windows,
)
from quantica.portfolio import (
    PortfolioConstraints,
    deflated_sharpe_ratio_from_trials,
    generate_trial_returns,
    minimum_variance_weights,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

from apps._data import load_ff_sample

_ANNUALISE = np.sqrt(12.0)


def covariance_comparison(
    *,
    train_window: int = 60,
    test_window: int = 12,
    n_random: int = 40,
    seed: int = 20240720,
) -> pd.DataFrame:
    """Out-of-sample bias and min-variance realised vol for the three estimators.

    Races sample / Ledoit--Wolf / factor covariance on the bundled 49-industry panel
    (walk-forward, no lookahead).

    Parameters
    ----------
    train_window : int, optional
        Length of each estimation (training) window in months (default 60).
    test_window : int, optional
        Length of each out-of-sample scoring window in months (default 12).
    n_random : int, optional
        Number of random test portfolios drawn per window (default 40).
    seed : int, optional
        Seed for the random-portfolio generator (default 20240720).

    Returns
    -------
    pandas.DataFrame
        One row per estimator with columns ``estimator``, ``random_bias`` (realised /
        forecast vol on random portfolios, ``≈1`` is calibrated), ``calibrated`` (share
        of forecasts within the calibrated band), ``min_var_vol`` (annualised realised
        OOS vol of the estimator's own minimum-variance portfolio), and ``min_var_bias``.
    """
    sample = load_ff_sample()
    comparison = compare_estimators(
        sample.industry_excess,
        (SampleCovariance(), LedoitWolfCovariance(), FactorCovariance()),
        train_window=train_window,
        test_window=test_window,
        factor_returns=sample.factor_returns,
        rng=np.random.default_rng(seed),
        n_random_portfolios=n_random,
    )
    mv_vol = comparison.mean_min_variance_vol()
    rows = []
    for name in comparison.estimator_names:
        rows.append(
            {
                "estimator": name,
                "random_bias": comparison.bias[name].mean,
                "calibrated": comparison.bias[name].fraction_calibrated(),
                "min_var_vol": mv_vol[name] * _ANNUALISE,
                "min_var_bias": comparison.min_variance_bias[name].mean,
            }
        )
    return pd.DataFrame(rows)


def jagannathan_ma(*, train_window: int = 60, test_window: int = 12) -> dict[str, object]:
    """The no-short-sale-is-shrinkage result on the bundled panel.

    Parameters
    ----------
    train_window : int, optional
        Length of each estimation window in months (default 60).
    test_window : int, optional
        Length of each out-of-sample scoring window in months (default 12).

    Returns
    -------
    dict
        Keys: ``table`` (a 2x2 ``pandas.DataFrame`` of realised vol, sample/ledoit-wolf
        by unconstrained/long-only), ``n_shorted`` and ``n_assets`` (shorted names and
        universe size on the first window), ``recovery_error`` (max abs error of the
        GMV of the Jagannathan--Ma shrunk covariance against the long-only weights),
        and ``cond_sample`` / ``cond_shrunk`` (condition numbers before/after shrinkage).
    """
    assets = load_ff_sample().industry_excess
    sample, lw = SampleCovariance(), LedoitWolfCovariance()
    long_only = PortfolioConstraints(long_only=True)
    windows = walk_forward_windows(assets.shape[0], train_window, test_window)

    def realised(weights_fn) -> float:  # type: ignore[no-untyped-def]
        vols = []
        for w in windows:
            train, test = assets[w.train_start : w.train_end], assets[w.test_start : w.test_end]
            vols.append(np.std(test @ weights_fn(train), ddof=1))
        return float(np.mean(vols)) * _ANNUALISE

    table = pd.DataFrame(
        {
            "covariance": ["sample", "ledoit-wolf"],
            "unconstrained": [
                realised(lambda tr: min_variance_weights(sample.estimate(tr))),
                realised(lambda tr: min_variance_weights(lw.estimate(tr))),
            ],
            "long_only": [
                realised(lambda tr: minimum_variance_weights(sample.estimate(tr), long_only)),
                realised(lambda tr: minimum_variance_weights(lw.estimate(tr), long_only)),
            ],
        }
    )

    # Exact Jagannathan--Ma equivalence on the first training window.
    cov = sample.estimate(assets[:train_window])
    w_lo = minimum_variance_weights(cov, long_only)
    lam = float(w_lo @ cov @ w_lo)
    mu = cov @ w_lo - lam
    cov_tilde = cov - (np.outer(mu, np.ones_like(mu)) + np.outer(np.ones_like(mu), mu))
    recovery_err = float(np.max(np.abs(min_variance_weights(cov_tilde) - w_lo)))
    return {
        "table": table,
        "n_shorted": int(np.sum(min_variance_weights(cov) < 0.0)),
        "n_assets": int(cov.shape[0]),
        "recovery_error": recovery_err,
        "cond_sample": condition_number(cov),
        "cond_shrunk": condition_number(cov_tilde),
    }


def overfit_search(
    *,
    n_periods: int = 360,
    n_trials: int = 100,
    planted_sharpe: float = 0.0,
    seed: int = 20240721,
    n_splits: int = 10,
) -> dict[str, object]:
    """Deflated-Sharpe / PBO verdict on an overfit search of many strategy trials.

    With ``planted_sharpe = 0`` every trial is pure noise (the winner is spurious);
    a positive ``planted_sharpe`` seeds one genuinely predictive trial.

    Parameters
    ----------
    n_periods : int, optional
        Number of return observations per trial (default 360).
    n_trials : int, optional
        Number of candidate strategies searched over (default 100).
    planted_sharpe : float, optional
        Per-period Sharpe planted into one trial; ``0`` leaves the matrix all-noise
        (default 0.0).
    seed : int, optional
        Seed for the trial-return generator (default 20240721).
    n_splits : int, optional
        Number of CSCV blocks for the PBO computation (default 10).

    Returns
    -------
    dict
        Keys: ``trial_sharpes`` (annualised per-trial Sharpes), ``selected`` and
        ``planted_index`` (the in-sample-best and the truly-planted trial), the
        annualised ``best_sharpe_ann`` and ``benchmark_sharpe_ann``, ``dsr`` and
        ``dsr_significant`` (deflated Sharpe and its 0.95 verdict), and ``pbo``.
    """
    trials = generate_trial_returns(
        n_periods, n_trials, np.random.default_rng(seed), planted_sharpe=planted_sharpe
    )
    dsr = deflated_sharpe_ratio_from_trials(trials.returns)
    pbo = probability_of_backtest_overfitting(trials.returns, n_splits=n_splits)
    trial_sharpes = (
        np.array([sharpe_ratio(trials.returns[:, j]) for j in range(n_trials)]) * _ANNUALISE
    )
    return {
        "trial_sharpes": trial_sharpes,
        "selected": dsr.selected,
        "planted_index": trials.planted_index,
        "best_sharpe_ann": dsr.observed_sr * _ANNUALISE,
        "benchmark_sharpe_ann": dsr.benchmark_sr * _ANNUALISE,
        "dsr": dsr.dsr,
        "dsr_significant": dsr.is_significant,
        "pbo": pbo.pbo,
    }
