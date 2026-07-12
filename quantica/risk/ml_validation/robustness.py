r"""Robustness — does the model behave when the inputs move?

Two SR 11-7 questions, quantified:

* :func:`prediction_stability` — **local smoothness**: perturb every input by a
  small, seeded Gaussian noise (scaled per feature) and measure how far the PDs
  move. Trees are step functions, so a gradient-boosting model can jump across a
  split boundary under a tiny perturbation — the tail of :math:`|\Delta PD|` is
  where that shows, and comparing it against the smooth logistic champion is the
  honest benchmark.
* :func:`performance_under_shift` — **degradation under covariate shift**:
  discrimination (AUC) and calibration (Hosmer--Lemeshow) evaluated on a
  development sample and on a shifted/monitoring sample, side by side. Uses
  ``dof = n_groups`` for the HL null because the scores are externally supplied
  relative to the evaluation samples (the model was not fitted on them) — see
  the credit package's size study for why the G-2 convention would over-reject.

Model access is a bare ``predict`` callable mapping ``(n, k)`` features to
``(n,)`` PDs — no internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.risk.credit.calibration import hosmer_lemeshow
from quantica.risk.credit.discrimination import auc

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantica.core.types import FloatArray

__all__ = [
    "PredictionStability",
    "ShiftDegradation",
    "performance_under_shift",
    "prediction_stability",
]

_DEFAULT_NOISE_SCALE = 0.01  # perturbation size in units of each feature's std
_DEFAULT_N_REPEATS = 10
_DEFAULT_HL_GROUPS = 10


@dataclass(frozen=True)
class PredictionStability:
    r"""Distribution of :math:`|\Delta PD|` under small input perturbations."""

    mean_abs_delta: float
    q95_abs_delta: float
    max_abs_delta: float
    noise_scale: float
    n_repeats: int
    n_rows: int


def prediction_stability(
    predict: Callable[[FloatArray], FloatArray],
    features: FloatArray,
    rng: np.random.Generator,
    *,
    noise_scale: float = _DEFAULT_NOISE_SCALE,
    n_repeats: int = _DEFAULT_N_REPEATS,
) -> PredictionStability:
    """Perturb inputs by seeded Gaussian noise and measure the PD movement.

    Each feature is perturbed by ``noise_scale`` times its own standard
    deviation, ``n_repeats`` times; the reported statistics pool all repeats.
    ``mean_abs_delta`` is the typical movement; ``max_abs_delta`` is where a
    step-function model betrays its split boundaries.
    """
    if noise_scale <= 0.0:
        raise ValueError(f"noise_scale must be positive, got {noise_scale}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("features must be a non-empty (n, k) matrix")
    stds = x.std(axis=0)
    baseline = np.asarray(predict(x), dtype=np.float64)
    deltas = []
    for _ in range(n_repeats):
        perturbed = x + noise_scale * stds * rng.standard_normal(x.shape)
        deltas.append(np.abs(np.asarray(predict(perturbed), dtype=np.float64) - baseline))
    pooled = np.concatenate(deltas)
    return PredictionStability(
        mean_abs_delta=float(pooled.mean()),
        q95_abs_delta=float(np.quantile(pooled, 0.95)),
        max_abs_delta=float(pooled.max()),
        noise_scale=noise_scale,
        n_repeats=n_repeats,
        n_rows=int(x.shape[0]),
    )


@dataclass(frozen=True)
class ShiftDegradation:
    """Discrimination and calibration, development vs shifted sample."""

    auc_dev: float
    auc_shift: float
    auc_delta: float
    hl_p_dev: float
    hl_p_shift: float

    @property
    def calibration_broke(self) -> bool:
        """Calibration held on development but is rejected on the shifted sample."""
        return self.hl_p_dev >= 0.05 > self.hl_p_shift


def performance_under_shift(
    y_dev: FloatArray,
    scores_dev: FloatArray,
    y_shift: FloatArray,
    scores_shift: FloatArray,
    *,
    n_groups: int = _DEFAULT_HL_GROUPS,
) -> ShiftDegradation:
    """AUC and Hosmer--Lemeshow on development vs shifted samples, side by side.

    The scores are externally supplied relative to both evaluation samples, so
    the HL null is :math:`\\chi^2` with ``n_groups`` degrees of freedom.
    """
    hl_dev = hosmer_lemeshow(y_dev, scores_dev, n_groups=n_groups, dof=n_groups)
    hl_shift = hosmer_lemeshow(y_shift, scores_shift, n_groups=n_groups, dof=n_groups)
    auc_dev = auc(y_dev, scores_dev)
    auc_shift = auc(y_shift, scores_shift)
    return ShiftDegradation(
        auc_dev=auc_dev,
        auc_shift=auc_shift,
        auc_delta=auc_shift - auc_dev,
        hl_p_dev=hl_dev.p_value,
        hl_p_shift=hl_shift.p_value,
    )
