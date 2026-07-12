"""Shared input validation for the credit-validation modules.

Every validator in this package consumes the same raw material — a binary
default indicator and a score/PD array — so the checks live once here.
"""

from __future__ import annotations

import numpy as np

from quantica.core.types import FloatArray


def validate_binary_scores(y: FloatArray, scores: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Validate and coerce a (default indicator, score) pair.

    ``y`` must be one-dimensional 0/1 with both classes present; ``scores`` must
    align with ``y`` and be finite. Returns float64 copies.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    s_arr = np.asarray(scores, dtype=np.float64)
    if y_arr.ndim != 1 or y_arr.size == 0:
        raise ValueError("y must be a non-empty 1-D array")
    if s_arr.shape != y_arr.shape:
        raise ValueError(f"scores shape {s_arr.shape} must match y shape {y_arr.shape}")
    if not np.all(np.isin(y_arr, (0.0, 1.0))):
        raise ValueError("y must contain only 0 (performing) and 1 (default)")
    if not (np.any(y_arr == 1.0) and np.any(y_arr == 0.0)):
        raise ValueError("y must contain both classes (at least one default and one performing)")
    if not np.all(np.isfinite(s_arr)):
        raise ValueError("scores must be finite")
    return y_arr, s_arr
