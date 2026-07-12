r"""Explainability checks validated — including validating SHAP itself.

- **Local accuracy** asserted for the exact explainers (TreeSHAP on the GBM,
  LinearSHAP on the logit) to near machine precision — and shown to *fail* when
  predictions are supplied on the wrong scale (probabilities against a log-odds
  explainer), the classic silent mistake the check exists to catch.
- **Driver recovery against the known DGP** (the centerpiece): the synthetic
  generator's total-contribution order is leverage > behavioural > profitability
  > liquidity > size; SHAP's global importance must reproduce it, for the GBM
  *and* the logistic champion.
- **Interaction recovery**: the planted leverage-times-behavioural interaction is
  the largest off-diagonal SHAP interaction, by a wide margin.
- **Direction**: attribution-vs-feature correlations carry the DGP signs, and
  leverage's is *attenuated* by the planted convexity (a U-shape weakens the
  linear correlation) — the honest nuance, asserted.
- **Rank stability** across explanation subsamples (fixed model): Spearman ≈ 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk.credit import generate_credit_portfolio
from quantica.risk.ml_validation import (
    attribution_direction,
    check_local_accuracy,
    driver_recovery,
    global_importance,
    rank_stability,
)

shap = pytest.importorskip("shap")
sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
sklearn_linear = pytest.importorskip("sklearn.linear_model")

_N = 30_000
_N_TRAIN = 18_000
_SEED = 42

# The generator's total log-odds contributions imply this driver order (main
# effects 0.9/-0.7/-0.5/-0.3/0.6 plus leverage's interaction 1.0 and convexity
# 0.55, and behavioural's interaction 1.0).
_EXPECTED_ORDER = ("leverage", "behavioural", "profitability", "liquidity", "size")


@pytest.fixture(scope="module")
def setup() -> dict[str, object]:
    sample = generate_credit_portfolio(_N, np.random.default_rng(_SEED))
    x_tr, y_tr = sample.features[:_N_TRAIN], sample.defaults[:_N_TRAIN]
    x_te = sample.features[_N_TRAIN:]
    gbm = sklearn_ensemble.HistGradientBoostingClassifier(random_state=0).fit(x_tr, y_tr)
    logit = sklearn_linear.LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
    explainer = shap.TreeExplainer(gbm)
    x_explain = x_te[:2000]
    return {
        "sample": sample,
        "gbm": gbm,
        "logit": logit,
        "explainer": explainer,
        "x_train": x_tr,
        "x_explain": x_explain,
        "shap_values": np.asarray(explainer.shap_values(x_explain)),
        "base_value": float(np.ravel(explainer.expected_value)[0]),
    }


# --------------------------------------------------------------------------- #
# 1. Local accuracy — validate the explainer itself
# --------------------------------------------------------------------------- #


def test_tree_shap_local_accuracy_on_log_odds(setup: dict[str, object]) -> None:
    gbm = setup["gbm"]
    result = check_local_accuracy(
        setup["shap_values"],  # type: ignore[arg-type]
        setup["base_value"],  # type: ignore[arg-type]
        gbm.decision_function(setup["x_explain"]),  # type: ignore[attr-defined]
    )
    assert result.passed
    assert result.max_abs_error < 1e-10  # exact TreeSHAP, float noise only


def test_local_accuracy_catches_the_wrong_output_scale(setup: dict[str, object]) -> None:
    # Probabilities against a log-odds explainer: additivity breaks, loudly.
    gbm = setup["gbm"]
    result = check_local_accuracy(
        setup["shap_values"],  # type: ignore[arg-type]
        setup["base_value"],  # type: ignore[arg-type]
        gbm.predict_proba(setup["x_explain"])[:, 1],  # type: ignore[attr-defined]
    )
    assert not result.passed
    assert result.max_abs_error > 1.0


def test_linear_shap_local_accuracy(setup: dict[str, object]) -> None:
    logit = setup["logit"]
    x_train = setup["x_train"]
    x = setup["x_explain"]
    explainer = shap.LinearExplainer(logit, x_train)
    sv = np.asarray(explainer.shap_values(x))
    base = float(np.ravel(explainer.expected_value)[0])
    result = check_local_accuracy(sv, base, logit.decision_function(x))  # type: ignore[attr-defined]
    assert result.passed


def test_local_accuracy_hand_example() -> None:
    sv = np.array([[0.5, -0.2], [1.0, 0.3]])
    preds = np.array([1.3, 2.3])  # base 1.0: rows sum to 0.3 and 1.3 exactly
    assert check_local_accuracy(sv, 1.0, preds).max_abs_error == pytest.approx(0.0, abs=1e-15)
    off = check_local_accuracy(sv, 1.0, preds + 0.5)
    assert off.max_abs_error == pytest.approx(0.5, abs=1e-15)
    assert not off.passed


# --------------------------------------------------------------------------- #
# 2. Driver recovery against the known data-generating process
# --------------------------------------------------------------------------- #


def test_gbm_shap_recovers_planted_driver_order(setup: dict[str, object]) -> None:
    sample = setup["sample"]
    importances = global_importance(
        setup["shap_values"],  # type: ignore[arg-type]
        sample.feature_names,  # type: ignore[attr-defined]
    )
    recovery = driver_recovery(importances, _EXPECTED_ORDER)
    assert recovery.exact_match  # the full planted order, recovered
    assert recovery.n_top_matched == len(_EXPECTED_ORDER)


def test_logit_shap_recovers_the_same_order(setup: dict[str, object]) -> None:
    logit = setup["logit"]
    sample = setup["sample"]
    explainer = shap.LinearExplainer(logit, setup["x_train"])
    sv = np.asarray(explainer.shap_values(setup["x_explain"]))
    importances = global_importance(sv, sample.feature_names)  # type: ignore[attr-defined]
    assert driver_recovery(importances, _EXPECTED_ORDER).exact_match


def test_planted_interaction_is_the_top_shap_interaction(setup: dict[str, object]) -> None:
    explainer = setup["explainer"]
    sample = setup["sample"]
    x = setup["x_explain"]
    inter = np.abs(np.asarray(explainer.shap_interaction_values(x[:500]))).mean(axis=0)  # type: ignore[attr-defined]
    off_diagonal = inter.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    i, j = np.unravel_index(np.argmax(off_diagonal), off_diagonal.shape)
    names = sample.feature_names  # type: ignore[attr-defined]
    assert {names[int(i)], names[int(j)]} == {"leverage", "behavioural"}
    # ... and by a wide margin over the next-largest off-diagonal pair.
    top = off_diagonal[i, j]
    off_diagonal[i, j] = off_diagonal[j, i] = 0.0
    assert top > 3.0 * off_diagonal.max()


def test_attribution_directions_match_dgp_signs(setup: dict[str, object]) -> None:
    corr = attribution_direction(
        setup["shap_values"],  # type: ignore[arg-type]
        setup["x_explain"],  # type: ignore[arg-type]
    )
    signs = np.sign(corr)
    np.testing.assert_array_equal(signs, [1.0, -1.0, -1.0, -1.0, 1.0])
    # The planted leverage convexity (U-shape) attenuates its linear correlation
    # relative to the purely monotone profitability effect — the honest nuance.
    assert abs(corr[0]) < abs(corr[1])


# --------------------------------------------------------------------------- #
# 3. Rank stability
# --------------------------------------------------------------------------- #


def test_importance_ranking_is_stable_across_subsamples(setup: dict[str, object]) -> None:
    explainer = setup["explainer"]
    sample = setup["sample"]
    x_te = sample.features[_N_TRAIN:]  # type: ignore[attr-defined]
    importances = np.array(
        [
            np.abs(np.asarray(explainer.shap_values(x_te[b * 2000 : (b + 1) * 2000]))).mean(  # type: ignore[attr-defined]
                axis=0
            )
            for b in range(5)
        ]
    )
    stability = rank_stability(importances)
    assert stability.min_spearman > 0.89
    assert stability.n_replications == 5


def test_rank_stability_detects_unstable_rankings() -> None:
    rng = np.random.default_rng(0)
    noise = rng.random((6, 5))  # unrelated rankings
    assert rank_stability(noise).mean_spearman < 0.5


# --------------------------------------------------------------------------- #
# 4. Wiring / validation
# --------------------------------------------------------------------------- #


def test_global_importance_hand_example() -> None:
    sv = np.array([[1.0, -0.5, 0.0], [-1.0, 0.5, 0.2]])
    imps = global_importance(sv, ("a", "b", "c"))
    assert [f.name for f in imps] == ["a", "b", "c"]
    assert imps[0].importance == pytest.approx(1.0)
    assert imps[1].importance == pytest.approx(0.5)
    assert [f.rank for f in imps] == [0, 1, 2]


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        check_local_accuracy(np.empty((0, 2)), 0.0, np.empty(0))
    with pytest.raises(ValueError, match="predictions shape"):
        check_local_accuracy(np.ones((3, 2)), 0.0, np.ones(2))
    with pytest.raises(ValueError, match="feature_names"):
        global_importance(np.ones((3, 2)), ("a", "b", "c"))
    imps = global_importance(np.ones((3, 2)), ("a", "b"))
    with pytest.raises(ValueError, match="not present"):
        driver_recovery(imps, ("a", "zz"))
    with pytest.raises(ValueError, match="B >= 2"):
        rank_stability(np.ones((1, 4)))
    with pytest.raises(ValueError, match="must match"):
        attribution_direction(np.ones((3, 2)), np.ones((3, 3)))
