r"""Champion / challenger on the synthetic book — the workflow the battery serves.

The synthetic generator plants an interaction and a convexity a linear logit
cannot represent, so out of sample the gradient-boosting challenger must
out-discriminate the logistic champion — and neither can beat the generative
true-PD score (an upper-bound sanity only synthetic data affords). The
*calibration* comparison is left to the report script: it is a finding to
present honestly, not an assertion to hard-code.

Requires scikit-learn (a dev/demo dependency — the validators themselves never
touch a model object).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk.credit import auc, generate_credit_portfolio, grade_calibration

sklearn_linear = pytest.importorskip("sklearn.linear_model")
sklearn_ensemble = pytest.importorskip("sklearn.ensemble")

_N_OBLIGORS = 30_000
_N_TRAIN = 18_000
_SEED = 42


@pytest.fixture(scope="module")
def fitted() -> dict[str, np.ndarray]:
    sample = generate_credit_portfolio(_N_OBLIGORS, np.random.default_rng(_SEED))
    x_train, y_train = sample.features[:_N_TRAIN], sample.defaults[:_N_TRAIN]
    x_test = sample.features[_N_TRAIN:]
    champion = sklearn_linear.LogisticRegression(max_iter=1000).fit(x_train, y_train)
    challenger = sklearn_ensemble.HistGradientBoostingClassifier(random_state=0).fit(
        x_train, y_train
    )
    return {
        "y": sample.defaults[_N_TRAIN:],
        "true_pd": sample.true_pd[_N_TRAIN:],
        "champion": champion.predict_proba(x_test)[:, 1],
        "challenger": challenger.predict_proba(x_test)[:, 1],
    }


def test_challenger_out_discriminates_champion_out_of_sample(
    fitted: dict[str, np.ndarray],
) -> None:
    auc_champion = auc(fitted["y"], fitted["champion"])
    auc_challenger = auc(fitted["y"], fitted["challenger"])
    # The planted nonlinearity is worth ~5 AUC points at this seed; require a
    # clear margin so the assertion is about the mechanism, not noise.
    assert auc_challenger > auc_champion + 0.02


def test_true_pd_is_the_discrimination_ceiling(fitted: dict[str, np.ndarray]) -> None:
    auc_truth = auc(fitted["y"], fitted["true_pd"])
    assert auc(fitted["y"], fitted["champion"]) < auc_truth
    assert auc(fitted["y"], fitted["challenger"]) < auc_truth


def test_grade_calibration_runs_on_fitted_model_output(
    fitted: dict[str, np.ndarray],
) -> None:
    rows = grade_calibration(fitted["y"], fitted["challenger"], n_grades=7)
    assert len(rows) == 7
    assert sum(r.n_obligors for r in rows) == _N_OBLIGORS - _N_TRAIN
    # Every grade's p-values are proper probabilities.
    assert all(0.0 <= r.binomial_p <= 1.0 and 0.0 <= r.jeffreys_p <= 1.0 for r in rows)
