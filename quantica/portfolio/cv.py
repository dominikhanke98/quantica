r"""Purged and embargoed cross-validation (López de Prado, AFML ch. 7).

Standard k-fold cross-validation is *wrong* for financial series with overlapping
labels. When the label at time :math:`t` spans :math:`[t, t+h]` (e.g. an
:math:`h`-period forward return), a training observation adjacent to the test set
shares label components with it — so the "out-of-sample" test is contaminated and
measured performance is inflated. Two corrections fix it:

* **Purging** — drop from the training set any observation whose label window
  overlaps the test block's label window (here, anything within ``label_horizon`` of
  the test block on *either* side, since overlapping labels leak both ways).
* **Embargo** — additionally drop a buffer of ``embargo`` observations immediately
  after the test block, to defend against serial correlation the fixed horizon does
  not capture.

:func:`purged_kfold_indices` returns ``(train, test)`` index arrays with these two
buffers applied; the walk-forward window machinery in
:func:`quantica.factor.evaluation.walk_forward_windows` already enforces the simpler
past-only no-lookahead property, and this module extends it to the k-fold /
overlapping-label setting the backtest-validity layer needs.

The headline validation is a **known-truth leakage test**: with overlapping labels,
predicting a test point from its temporally nearest training neighbour shows spurious
skill *without* purging and none *with* it — the leakage is real and purging removes
it.

References
----------
López de Prado, M. (2018), *Advances in Financial Machine Learning*, ch. 7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import IntArray

__all__ = [
    "PurgedFold",
    "purged_kfold_indices",
]


class PurgedFold(NamedTuple):
    """One purged/embargoed cross-validation fold."""

    train: IntArray
    test: IntArray


def purged_kfold_indices(
    n_obs: int,
    n_splits: int,
    *,
    label_horizon: int = 0,
    embargo: int = 0,
) -> tuple[PurgedFold, ...]:
    r"""Purged, embargoed k-fold train/test index splits over ``n_obs`` observations.

    The test folds are contiguous blocks tiling ``0..n_obs-1``. For each, the training
    set is everything else with two buffers removed: a two-sided **purge** of
    ``label_horizon`` observations around the block (labels overlap both ways), and a
    trailing **embargo** of ``embargo`` further observations.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    n_splits : int
        Number of folds (contiguous test blocks); must be in ``2..n_obs``.
    label_horizon : int
        Label span :math:`h`; observations within ``h`` of a test block (either side)
        are purged from training. ``0`` means non-overlapping labels (no purge).
    embargo : int
        Extra observations dropped immediately after each test block.

    Returns
    -------
    tuple of PurgedFold
        One ``(train, test)`` pair per fold, as integer index arrays.
    """
    if n_splits < 2 or n_splits > n_obs:
        raise ValueError(f"n_splits must be in 2..n_obs ({n_obs}), got {n_splits}")
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")

    indices = np.arange(n_obs)
    blocks = np.array_split(indices, n_splits)
    folds: list[PurgedFold] = []
    for block in blocks:
        t0, t1 = int(block[0]), int(block[-1])  # inclusive test bounds
        train_mask = np.ones(n_obs, dtype=bool)
        # Remove the test block plus a two-sided purge of `label_horizon`.
        purge_lo = max(0, t0 - label_horizon)
        purge_hi = min(n_obs - 1, t1 + label_horizon)
        train_mask[purge_lo : purge_hi + 1] = False
        # Trailing embargo immediately after the purged-after region.
        embargo_hi = min(n_obs, purge_hi + 1 + embargo)
        train_mask[purge_hi + 1 : embargo_hi] = False
        folds.append(
            PurgedFold(
                train=indices[train_mask].astype(np.intp),
                test=block.astype(np.intp),
            )
        )
    return tuple(folds)
