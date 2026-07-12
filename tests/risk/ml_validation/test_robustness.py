r"""Robustness checks validated — with an analytic anchor for the metric itself.

- **Analytic anchor**: for a *linear* predictor :math:`f(x) = w \cdot x`, the
  perturbation response has a closed form —
  :math:`\mathbb E|\Delta f| = \sqrt{2/\pi}\,\sigma_\Delta` with
  :math:`\sigma_\Delta^2 = \sum_j w_j^2 (\text{noise}\cdot s_j)^2` — so the
  stability metric is validated against a known value before it is trusted on
  the GBM (validate the validator, again).
- **The honest finding**: the GBM's tail instability. Trees are step functions;
  under 1% input noise the GBM's worst-case :math:`|\Delta PD|` is an order of
  magnitude beyond the logistic champion's, even though its *mean* movement is
  modest. Asserted, and fed to the recommendation.
- **Shift degradation** wiring equals direct AUC / Hosmer--Lemeshow calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.core.types import FloatArray
from quantica.risk.credit import generate_credit_portfolio
from quantica.risk.credit.calibration import hosmer_lemeshow
from quantica.risk.credit.discrimination import auc
from quantica.risk.ml_validation import (
    ShiftDegradation,
    performance_under_shift,
    prediction_stability,
)

sklearn_ensemble = pytest.importorskip("sklearn.ensemble")
sklearn_linear = pytest.importorskip("sklearn.linear_model")


# --------------------------------------------------------------------------- #
# 1. Analytic anchor for the stability metric
# --------------------------------------------------------------------------- #


def test_prediction_stability_matches_linear_closed_form() -> None:
    rng = np.random.default_rng(0)
    w = np.array([0.8, -0.5, 0.3])
    stds = np.array([1.0, 2.0, 0.5])
    x = rng.standard_normal((20_000, 3)) * stds

    def predict(features: FloatArray) -> FloatArray:
        return np.asarray(features @ w, dtype=np.float64)

    noise = 0.05
    result = prediction_stability(predict, x, np.random.default_rng(1), noise_scale=noise)
    sigma_delta = noise * float(np.sqrt(np.sum((w * x.std(axis=0)) ** 2)))
    expected_mean = float(np.sqrt(2.0 / np.pi)) * sigma_delta
    assert result.mean_abs_delta == pytest.approx(expected_mean, rel=0.02)


def test_prediction_stability_is_seeded() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((500, 2))

    def predict(features: FloatArray) -> FloatArray:
        return np.asarray(features.sum(axis=1), dtype=np.float64)

    a = prediction_stability(predict, x, np.random.default_rng(3))
    b = prediction_stability(predict, x, np.random.default_rng(3))
    assert a == b


# --------------------------------------------------------------------------- #
# 2. The honest finding: GBM tail instability vs the smooth champion
# --------------------------------------------------------------------------- #


def test_gbm_is_less_stable_than_logit_especially_in_the_tail() -> None:
    sample = generate_credit_portfolio(20_000, np.random.default_rng(42))
    x_tr, y_tr = sample.features[:14_000], sample.defaults[:14_000]
    x_te = sample.features[14_000:]
    gbm = sklearn_ensemble.HistGradientBoostingClassifier(random_state=0).fit(x_tr, y_tr)
    logit = sklearn_linear.LogisticRegression(max_iter=1000).fit(x_tr, y_tr)

    def pd_of(model):  # type: ignore[no-untyped-def]
        def predict(features: FloatArray) -> FloatArray:
            return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)

        return predict

    gbm_stab = prediction_stability(pd_of(gbm), x_te, np.random.default_rng(0), n_repeats=5)
    logit_stab = prediction_stability(pd_of(logit), x_te, np.random.default_rng(0), n_repeats=5)
    # Mean movement: GBM materially worse. Tail: an order of magnitude worse —
    # a tiny perturbation can cross a split boundary and jump the PD.
    assert gbm_stab.mean_abs_delta > 2.0 * logit_stab.mean_abs_delta
    assert gbm_stab.max_abs_delta > 5.0 * logit_stab.max_abs_delta


# --------------------------------------------------------------------------- #
# 3. Shift degradation
# --------------------------------------------------------------------------- #


def test_performance_under_shift_equals_direct_calls() -> None:
    rng = np.random.default_rng(4)
    p_dev = 1.0 / (1.0 + np.exp(-rng.normal(-3.0, 1.0, 6000)))
    y_dev = (rng.random(6000) < p_dev).astype(float)
    p_shift = 1.0 / (1.0 + np.exp(-rng.normal(-2.5, 1.0, 6000)))
    y_shift = (rng.random(6000) < p_shift).astype(float)
    result = performance_under_shift(y_dev, p_dev, y_shift, p_shift)
    assert result.auc_dev == pytest.approx(auc(y_dev, p_dev))
    assert result.auc_shift == pytest.approx(auc(y_shift, p_shift))
    assert result.auc_delta == pytest.approx(result.auc_shift - result.auc_dev)
    assert result.hl_p_dev == pytest.approx(hosmer_lemeshow(y_dev, p_dev, dof=10).p_value)
    assert result.hl_p_shift == pytest.approx(hosmer_lemeshow(y_shift, p_shift, dof=10).p_value)


def test_calibration_broke_property() -> None:
    broke = ShiftDegradation(
        auc_dev=0.9, auc_shift=0.88, auc_delta=-0.02, hl_p_dev=0.4, hl_p_shift=0.001
    )
    held = ShiftDegradation(
        auc_dev=0.9, auc_shift=0.88, auc_delta=-0.02, hl_p_dev=0.4, hl_p_shift=0.2
    )
    assert broke.calibration_broke
    assert not held.calibration_broke


def test_input_validation() -> None:
    def predict(features: FloatArray) -> FloatArray:
        return np.asarray(features.sum(axis=1), dtype=np.float64)

    x = np.ones((5, 2))
    with pytest.raises(ValueError, match="noise_scale"):
        prediction_stability(predict, x, np.random.default_rng(0), noise_scale=0.0)
    with pytest.raises(ValueError, match="n_repeats"):
        prediction_stability(predict, x, np.random.default_rng(0), n_repeats=0)
    with pytest.raises(ValueError, match="non-empty"):
        prediction_stability(predict, np.empty((0, 2)), np.random.default_rng(0))
