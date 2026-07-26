"""Validation of the Markov regime-switching model (numerical-validation skill).

The headline is **known-truth state recovery**: simulate from a *known* 2-state Markov-switching
process, then confirm the Hamilton filter + EM recover the true means/variances/transition matrix
*and* that the smoothed probabilities classify each observation into the regime it was actually
generated from. A machine-precision anchor cross-checks the filtered/smoothed probabilities and the
log-likelihood against ``statsmodels``' ``MarkovRegression`` at identical parameters, and the EM
required property — a **non-decreasing log-likelihood** — is asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.timeseries import (
    fit_markov_switching,
    hamilton_filter,
    kim_smoother,
    simulate_markov_switching,
)

_TRUE_MEANS = np.array([0.05, -0.10])
_TRUE_VARIANCES = np.array([1.0, 9.0])  # calm vs crisis
_TRUE_P = np.array([[0.97, 0.03], [0.10, 0.90]])  # persistent regimes


# --------------------------------------------------------------------------- #
# The headline: known-truth recovery of parameters and hidden states
# --------------------------------------------------------------------------- #


def test_known_truth_parameters_and_states_recovered() -> None:
    """The EM recovers the true regime parameters and classifies the hidden states well."""
    y, states = simulate_markov_switching(
        3500, _TRUE_MEANS, _TRUE_VARIANCES, _TRUE_P, rng=np.random.default_rng(3)
    )
    fit = fit_markov_switching(y, 2, n_starts=6, rng=np.random.default_rng(0))

    # Variances (regimes sorted ascending, so 0 = calm, 1 = crisis, matching the simulation).
    assert np.allclose(fit.variances, _TRUE_VARIANCES, rtol=0.20)
    assert np.allclose(fit.transition_matrix, _TRUE_P, atol=0.05)
    # The planted hidden states are recovered by the smoothed probabilities.
    accuracy = (fit.most_likely_states() == states).mean()
    assert accuracy > 0.85


def test_switching_variance_only_recovers_regimes() -> None:
    """With a common mean (switching_variance model) the two vol regimes are still recovered."""
    means = np.array([0.0, 0.0])
    y, states = simulate_markov_switching(
        3000, means, _TRUE_VARIANCES, _TRUE_P, rng=np.random.default_rng(5)
    )
    fit = fit_markov_switching(y, 2, switching_mean=False, n_starts=6, rng=np.random.default_rng(1))
    assert np.allclose(fit.variances, _TRUE_VARIANCES, rtol=0.20)
    assert fit.means[0] == fit.means[1]  # the mean does not switch
    assert (fit.most_likely_states() == states).mean() > 0.85


# --------------------------------------------------------------------------- #
# The anchor: filtered/smoothed probabilities vs statsmodels at fixed parameters
# --------------------------------------------------------------------------- #


def test_filter_and_smoother_match_statsmodels() -> None:
    """At identical parameters the filter, smoother and log-likelihood match statsmodels exactly."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    rng = np.random.default_rng(1)
    y = np.concatenate(
        [rng.normal(0.0, 1.0, 150), rng.normal(0.5, 3.0, 150), rng.normal(0.0, 1.0, 100)]
    )
    means, variances = np.array([0.0, 0.5]), np.array([1.0, 9.0])
    p_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])

    result = hamilton_filter(y, means, variances, p_matrix)
    smoothed = kim_smoother(result.filtered, result.predicted, p_matrix)

    model = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    # statsmodels params: [p[0->0], p[1->0], const[0], const[1], sigma2[0], sigma2[1]].
    reference = model.smooth([0.95, 0.05, 0.0, 0.5, 1.0, 9.0])
    assert np.isclose(result.loglikelihood, reference.llf, atol=1e-9)
    assert np.allclose(
        result.filtered[:, 0], reference.filtered_marginal_probabilities[:, 0], atol=1e-10
    )
    assert np.allclose(smoothed[:, 0], reference.smoothed_marginal_probabilities[:, 0], atol=1e-10)


# --------------------------------------------------------------------------- #
# EM sanity and algorithm properties
# --------------------------------------------------------------------------- #


def test_em_loglikelihood_is_monotonically_non_decreasing() -> None:
    """The EM log-likelihood never decreases across iterations (a required property)."""
    y, _states = simulate_markov_switching(
        2000, _TRUE_MEANS, _TRUE_VARIANCES, _TRUE_P, rng=np.random.default_rng(7)
    )
    fit = fit_markov_switching(y, 2, n_starts=4, rng=np.random.default_rng(2))
    steps = np.diff(fit.loglikelihood_history)
    assert np.all(steps >= -1e-6)  # non-decreasing up to floating-point noise


def test_regimes_are_ordered_and_label_switching_resolved() -> None:
    """Regimes come back sorted by variance from any start, so labels are stable."""
    y, _states = simulate_markov_switching(
        2500, _TRUE_MEANS, _TRUE_VARIANCES, _TRUE_P, rng=np.random.default_rng(9)
    )
    fit_a = fit_markov_switching(y, 2, n_starts=5, rng=np.random.default_rng(10))
    fit_b = fit_markov_switching(y, 2, n_starts=5, rng=np.random.default_rng(11))
    assert fit_a.variances[0] < fit_a.variances[1]  # calm before crisis
    assert np.allclose(fit_a.variances, fit_b.variances, rtol=0.05)  # start-independent


def test_probabilities_are_valid_distributions() -> None:
    """Filtered and smoothed probabilities are non-negative and sum to one across regimes."""
    y, _states = simulate_markov_switching(
        1500, _TRUE_MEANS, _TRUE_VARIANCES, _TRUE_P, rng=np.random.default_rng(12)
    )
    fit = fit_markov_switching(y, 2, n_starts=4, rng=np.random.default_rng(3))
    for probs in (fit.filtered_probabilities, fit.smoothed_probabilities):
        assert np.all(probs >= 0.0)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-10)


def test_expected_durations_match_persistence() -> None:
    """Expected regime durations equal 1/(1 - P_kk) for the fitted transition matrix."""
    y, _states = simulate_markov_switching(
        3000, _TRUE_MEANS, _TRUE_VARIANCES, _TRUE_P, rng=np.random.default_rng(13)
    )
    fit = fit_markov_switching(y, 2, n_starts=3, rng=np.random.default_rng(4))
    expected = 1.0 / (1.0 - np.diag(fit.transition_matrix))
    assert np.allclose(fit.expected_durations(), expected)
    assert fit.expected_durations()[0] > fit.expected_durations()[1]  # calm is more persistent


def test_smoother_endpoint_equals_filter() -> None:
    """The smoothed distribution at the final observation equals the filtered one there."""
    rng = np.random.default_rng(14)
    y = rng.normal(0.0, 1.0, 200)
    means, variances = np.array([-0.5, 0.5]), np.array([1.0, 4.0])
    p_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = hamilton_filter(y, means, variances, p_matrix)
    smoothed = kim_smoother(result.filtered, result.predicted, p_matrix)
    assert np.allclose(smoothed[-1], result.filtered[-1])


def test_invalid_arguments_are_rejected() -> None:
    """Bad regime counts, start counts and simulation dimensions raise clear errors."""
    rng = np.random.default_rng(15)
    y = rng.normal(size=100)
    with pytest.raises(ValueError, match="k_regimes must be at least 2"):
        fit_markov_switching(y, 1, rng=rng)
    with pytest.raises(ValueError, match="n_starts must be at least 1"):
        fit_markov_switching(y, 2, n_starts=0, rng=rng)
    with pytest.raises(ValueError, match="dimensions must agree"):
        simulate_markov_switching(
            100, np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.eye(3), rng=rng
        )
