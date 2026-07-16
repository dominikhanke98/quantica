"""Validation of purged + embargoed cross-validation.

Structural checks pin the index bookkeeping (test folds tile the timeline once,
train/test never overlap, the purge and embargo remove exactly the right bands). The
headline is a **known-truth leakage test**: with deliberately overlapping labels, a
model that peeks at its temporally nearest training neighbour shows spurious skill
*without* purging and essentially none *with* it — proving the leakage is real and
that purging removes it.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.portfolio.cv import purged_kfold_indices

# --------------------------------------------------------------------------- #
# Structural correctness
# --------------------------------------------------------------------------- #


def test_test_folds_tile_the_timeline_once() -> None:
    folds = purged_kfold_indices(100, n_splits=5)
    all_test = np.concatenate([f.test for f in folds])
    assert np.array_equal(np.sort(all_test), np.arange(100))
    assert len(all_test) == len(set(all_test.tolist()))  # no period tested twice


def test_train_and_test_are_disjoint() -> None:
    for f in purged_kfold_indices(100, n_splits=5, label_horizon=3, embargo=2):
        assert set(f.train.tolist()).isdisjoint(f.test.tolist())


def test_no_purge_no_embargo_is_plain_complement() -> None:
    folds = purged_kfold_indices(50, n_splits=5)
    for f in folds:
        expected_train = np.setdiff1d(np.arange(50), f.test)
        assert np.array_equal(f.train, expected_train)


def test_purge_removes_label_horizon_on_both_sides() -> None:
    """A middle fold loses `label_horizon` training points on each side."""
    folds = purged_kfold_indices(100, n_splits=10, label_horizon=4)
    middle = folds[5]
    t0, t1 = int(middle.test[0]), int(middle.test[-1])
    train = set(middle.train.tolist())
    # The 4 indices immediately before and after the block are purged.
    for gap in range(1, 5):
        assert (t0 - gap) not in train
        assert (t1 + gap) not in train
    # The 5th index out on each side survives.
    assert (t0 - 5) in train
    assert (t1 + 5) in train


def test_embargo_removes_trailing_points_only() -> None:
    """Embargo drops points after the (purged) block, not before it."""
    folds = purged_kfold_indices(100, n_splits=10, label_horizon=0, embargo=3)
    middle = folds[5]
    t0, t1 = int(middle.test[0]), int(middle.test[-1])
    train = set(middle.train.tolist())
    for gap in range(1, 4):
        assert (t1 + gap) not in train  # embargoed after
    assert (t1 + 4) in train
    assert (t0 - 1) in train  # nothing removed before when horizon is 0


def test_rejects_bad_split_count() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        purged_kfold_indices(10, n_splits=1)
    with pytest.raises(ValueError, match="n_splits"):
        purged_kfold_indices(10, n_splits=11)


def test_rejects_negative_buffers() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        purged_kfold_indices(50, n_splits=5, label_horizon=-1)


# --------------------------------------------------------------------------- #
# The headline: known-truth leakage from overlapping labels
# --------------------------------------------------------------------------- #


def _nearest_neighbour_leakage_skill(labels: np.ndarray, folds: tuple) -> float:
    """Correlation between each test label and its nearest *training* label.

    With overlapping labels, an adjacent (lag-1) training neighbour shares label
    components with a test point, so this correlation is spuriously high unless the
    neighbour has been purged far enough away that the labels no longer overlap.
    """
    preds: list[float] = []
    actuals: list[float] = []
    for fold in folds:
        train_idx = fold.train
        for j in fold.test:
            nearest = train_idx[np.argmin(np.abs(train_idx - j))]
            preds.append(float(labels[nearest]))
            actuals.append(float(labels[j]))
    return float(np.corrcoef(preds, actuals)[0, 1])


def test_purging_removes_overlapping_label_leakage() -> None:
    """Without purging the nearest-neighbour 'skill' is high; purging kills it."""
    rng = np.random.default_rng(0)
    horizon = 10
    base = rng.normal(0.0, 1.0, size=600 + horizon)
    # Overlapping labels: h-period forward cumulative sums (strongly autocorrelated).
    labels = np.array([base[t : t + horizon].sum() for t in range(600)])

    # Small folds so every test point sits near a training boundary (where overlap
    # leaks); a contiguous block's interior would dilute the effect.
    unpurged = purged_kfold_indices(600, n_splits=100, label_horizon=0, embargo=0)
    purged = purged_kfold_indices(600, n_splits=100, label_horizon=horizon, embargo=horizon)

    leaked = _nearest_neighbour_leakage_skill(labels, unpurged)
    cleaned = _nearest_neighbour_leakage_skill(labels, purged)

    assert leaked > 0.7  # strong spurious skill from adjacent overlapping labels
    assert abs(cleaned) < 0.15  # purging pushes the nearest neighbour past the overlap
    assert cleaned < leaked - 0.5  # and the gap is large
