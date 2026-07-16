r"""The backtest-validity layer — the headline deliverable.

Everyone ships a backtester; this ships the *test of whether the backtest means
anything*. A high in-sample Sharpe ratio is nearly free: search enough strategies on
enough noise and one will look brilliant by chance. This module implements the
statistics that price that selection in — the model-validation discipline of the rest
of the repo, now applied to strategy backtests.

The four instruments:

* :func:`probabilistic_sharpe_ratio` (PSR, Bailey & López de Prado 2012) — the
  probability that the *true* Sharpe exceeds a benchmark, correcting the Sharpe
  estimator's standard error for track-record length and return **non-normality**
  (skew and kurtosis inflate the estimator's variance).
* :func:`deflated_sharpe_ratio` (DSR, Bailey & López de Prado 2014) — PSR with the
  benchmark set to the *expected maximum* Sharpe of :math:`N` trials under the null
  of no skill. This is the single most important correction: it turns "best of many"
  from a virtue into the null hypothesis it actually is.
* :func:`probability_of_backtest_overfitting` (PBO via CSCV, Bailey et al. 2017) —
  combinatorially-symmetric cross-validation. Across every balanced train/test split
  of the trial matrix, how often is the in-sample-best strategy *below median* out of
  sample? A high PBO means the selection procedure itself overfits.
* :func:`minimum_track_record_length` (MinTRL) — how long a track record must be for
  an observed Sharpe to be significant at a confidence level.

Everything is computed in **per-period** Sharpe units (annualise only for display),
and leans on ``numpy``/``scipy`` for the statistics — the deliverable is the validity
framework, not the estimators.

References
----------
Bailey, D. & López de Prado, M. (2012), "The Sharpe ratio efficient frontier",
*Journal of Risk*.
Bailey, D. & López de Prado, M. (2014), "The deflated Sharpe ratio", *Journal of
Portfolio Management*.
Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017), "The probability of
backtest overfitting", *Journal of Computational Finance*.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kurtosis, norm, rankdata, skew

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "DeflatedSharpeResult",
    "PBOResult",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_trials",
    "expected_maximum_sharpe",
    "minimum_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_ratio",
]

#: Euler--Mascheroni constant, used in the expected-maximum-Sharpe formula.
_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: FloatArray, *, riskfree: float = 0.0, periods_per_year: int = 1) -> float:
    r"""Sharpe ratio of a return series (per-period by default).

    ``periods_per_year > 1`` annualises by :math:`\sqrt{\text{periods}}`. The validity
    statistics below all use the **per-period** ratio; annualise only for display.
    """
    excess = np.asarray(returns, dtype=np.float64) - riskfree
    sd = float(np.std(excess, ddof=1))
    # Treat a series that is constant to within floating-point precision (relative to
    # its own scale) as zero-variance — its Sharpe is undefined, reported as 0.
    scale = float(np.max(np.abs(excess))) if excess.size else 0.0
    if sd <= np.finfo(np.float64).eps * max(scale, 1.0):
        return 0.0
    return float(float(np.mean(excess)) / sd * np.sqrt(periods_per_year))


def _sharpe_std_error_factor(observed_sr: float, skewness: float, kurt: float) -> float:
    r"""The variance-inflation term :math:`1 - \gamma_3 SR + \tfrac{\gamma_4-1}{4} SR^2`.

    ``kurt`` is the **non-excess** kurtosis (3 for a normal). This is the numerator of
    the Sharpe estimator's variance (Mertens 2002 / Lo 2002); it exceeds 1 for the
    negative-skew, fat-tailed returns typical of real strategies, so ignoring it
    *overstates* significance.
    """
    return 1.0 - skewness * observed_sr + 0.25 * (kurt - 1.0) * observed_sr**2


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    *,
    benchmark_sr: float = 0.0,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    r"""Probability that the true Sharpe exceeds ``benchmark_sr`` (Bailey--LdP 2012).

    .. math::

        \mathrm{PSR}(SR^*) = \Phi\!\left(
            \frac{(\widehat{SR} - SR^*)\,\sqrt{n-1}}
                 {\sqrt{1 - \gamma_3 \widehat{SR} + \tfrac{\gamma_4-1}{4}\widehat{SR}^2}}
        \right)

    with all Sharpe ratios in the same per-period units, ``kurt`` the non-excess
    kurtosis. Returns exactly ``0.5`` when the observed equals the benchmark.
    """
    if n_obs < 2:
        raise ValueError("PSR needs at least 2 observations")
    var_factor = _sharpe_std_error_factor(observed_sr, skew, kurt)
    if var_factor <= 0.0:
        raise ValueError("degenerate Sharpe variance factor (implausible moments)")
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / np.sqrt(var_factor)
    return float(norm.cdf(z))


def expected_maximum_sharpe(n_trials: int, sr_variance: float) -> float:
    r"""Expected maximum Sharpe of ``n_trials`` independent no-skill strategies.

    .. math::

        \mathbb E[\max_n \widehat{SR}_n] \approx
        \sqrt{V}\left[(1-\gamma)\,\Phi^{-1}\!\Big(1-\tfrac1N\Big)
                      + \gamma\,\Phi^{-1}\!\Big(1-\tfrac1{N e}\Big)\right],

    where :math:`V` is the cross-trial variance of the Sharpe estimates and
    :math:`\gamma` is the Euler--Mascheroni constant. This is the benchmark the
    deflated Sharpe ratio deflates against — the Sharpe you would expect to see from
    the *luckiest* of ``n_trials`` even when none has any edge.
    """
    if n_trials < 2:
        raise ValueError("need at least 2 trials")
    if sr_variance < 0.0:
        raise ValueError("sr_variance must be non-negative")
    z1 = float(norm.ppf(1.0 - 1.0 / n_trials))
    z2 = float(norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    return float(np.sqrt(sr_variance) * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_obs: int,
    n_trials: int,
    sr_variance: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    r"""Deflated Sharpe ratio: PSR against the expected-maximum-Sharpe benchmark.

    Combines :func:`expected_maximum_sharpe` (the multiple-testing correction) with
    :func:`probabilistic_sharpe_ratio` (the length / non-normality correction). A DSR
    near 1 means the strategy is significant *after* accounting for the number of
    trials; a DSR near 0.5 or below means the observed Sharpe is no better than the
    luckiest coin-flip among ``n_trials``.
    """
    benchmark = expected_maximum_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(
        observed_sr, n_obs, benchmark_sr=benchmark, skew=skew, kurt=kurt
    )


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """The deflated Sharpe ratio of the selected trial, with its inputs exposed."""

    dsr: float
    selected: int
    observed_sr: float
    benchmark_sr: float
    n_trials: int
    n_obs: int

    @property
    def is_significant(self) -> bool:
        """Whether the deflated Sharpe clears the conventional 0.95 threshold."""
        return self.dsr >= 0.95


def deflated_sharpe_ratio_from_trials(
    trial_returns: FloatArray, *, riskfree: float = 0.0
) -> DeflatedSharpeResult:
    r"""Deflate the best strategy in a ``(T, N)`` trial-return matrix.

    Picks the highest-Sharpe column, then deflates *its* Sharpe for the ``N`` trials
    (variance taken across the trial Sharpes) and for its own return moments. This is
    the operational headline: run it on a matrix of pure-noise trials and the best one
    is flagged insignificant; plant a real signal and it survives.
    """
    r = np.asarray(trial_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("trial_returns must be 2-D (T, n_trials)")
    n_obs, n_trials = r.shape
    if n_trials < 2:
        raise ValueError("need at least 2 trials")
    srs = np.array([sharpe_ratio(r[:, j], riskfree=riskfree) for j in range(n_trials)])
    selected = int(np.argmax(srs))
    chosen = r[:, selected] - riskfree
    dsr = deflated_sharpe_ratio(
        float(srs[selected]),
        n_obs=n_obs,
        n_trials=n_trials,
        sr_variance=float(np.var(srs, ddof=1)),
        skew=float(skew(chosen)),
        kurt=float(kurtosis(chosen, fisher=False)),
    )
    benchmark = expected_maximum_sharpe(n_trials, float(np.var(srs, ddof=1)))
    return DeflatedSharpeResult(
        dsr=dsr,
        selected=selected,
        observed_sr=float(srs[selected]),
        benchmark_sr=benchmark,
        n_trials=n_trials,
        n_obs=n_obs,
    )


def minimum_track_record_length(
    observed_sr: float,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    r"""Minimum track-record length for the Sharpe to be significant (Bailey--LdP).

    The number of observations ``n`` at which :func:`probabilistic_sharpe_ratio`
    equals ``confidence``:

    .. math::

        \mathrm{MinTRL} = 1 + \Big(1 - \gamma_3 SR + \tfrac{\gamma_4-1}{4}SR^2\Big)
                              \left(\frac{Z_\alpha}{SR - SR^*}\right)^2.

    Requires ``observed_sr > benchmark_sr`` (otherwise no finite length suffices).
    """
    if observed_sr <= benchmark_sr:
        raise ValueError("observed_sr must exceed benchmark_sr for a finite MinTRL")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    var_factor = _sharpe_std_error_factor(observed_sr, skew, kurt)
    z_alpha = float(norm.ppf(confidence))
    return 1.0 + var_factor * (z_alpha / (observed_sr - benchmark_sr)) ** 2


@dataclass(frozen=True)
class PBOResult:
    """Probability of backtest overfitting and the CSCV logit distribution."""

    pbo: float
    #: Logit of the OOS relative rank of the IS-best strategy, one per CSCV split.
    logits: FloatArray
    n_splits: int
    n_combinations: int

    @property
    def median_logit(self) -> float:
        return float(np.median(self.logits))


def _column_sharpes(matrix: FloatArray) -> FloatArray:
    """Per-column (per-strategy) per-period Sharpe over the rows of ``matrix``."""
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0.0, mean / sd, 0.0)
    return np.asarray(sr, dtype=np.float64)


def probability_of_backtest_overfitting(
    trial_returns: FloatArray, *, n_splits: int = 16
) -> PBOResult:
    r"""Probability of backtest overfitting via combinatorially-symmetric CV.

    Partitions the ``T`` rows of the ``(T, N)`` trial matrix into ``n_splits``
    contiguous blocks, then over every balanced split (half the blocks in-sample, half
    out) records the out-of-sample relative rank :math:`\omega` of the strategy that
    was best in-sample, as a logit :math:`\log[\omega/(1-\omega)]`. **PBO is the
    fraction of splits where that logit is** :math:`\le 0` — i.e. the in-sample
    champion lands at or below the OOS median. Pure-noise trials give PBO ≈ 0.5 (the
    selection is worthless); a genuine edge gives PBO ≈ 0.

    ``n_splits`` must be even (balanced halves); the number of splits evaluated is
    :math:`\binom{S}{S/2}`.
    """
    r = np.asarray(trial_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("trial_returns must be 2-D (T, n_trials)")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 2")
    n_obs, n_trials = r.shape
    if n_trials < 2:
        raise ValueError("need at least 2 trials to rank")
    if n_obs < n_splits:
        raise ValueError("need at least n_splits observations")

    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits: list[float] = []
    all_block_ids = set(range(n_splits))
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = all_block_ids - set(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in sorted(oos_blocks)])
        is_sr = _column_sharpes(r[is_idx])
        oos_sr = _column_sharpes(r[oos_idx])
        best = int(np.argmax(is_sr))
        # Relative OOS rank of the IS-best (1 = worst .. N = best), ties averaged.
        oos_rank = float(rankdata(oos_sr)[best])
        omega = oos_rank / (n_trials + 1.0)
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)  # keep the logit finite
        logits.append(float(np.log(omega / (1.0 - omega))))

    logit_arr = np.array(logits, dtype=np.float64)
    return PBOResult(
        pbo=float(np.mean(logit_arr <= 0.0)),
        logits=logit_arr,
        n_splits=n_splits,
        n_combinations=len(logits),
    )
