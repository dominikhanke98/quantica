r"""Synthetic factor data with a *known* data-generating process.

Real factor data (Fama--French factors + an asset universe) is pulled by
``scripts/fetch_ff_factors.py`` and cached locally, but CI must never depend on a
network fetch — so the deterministic tests run on this generator, whose betas,
alphas and specific variances are **planted**. Recovering them from a fit is the
headline correctness check (the same known-truth discipline as the rest of the
risk pillar), and only synthetic data makes it possible.

The model is the textbook linear factor structure

.. math::

    r_{t,i} = \alpha_i + \sum_j \beta_{ij} f_{t,j} + \varepsilon_{t,i},
    \qquad \varepsilon_{t,i} \sim \mathcal N(0, \sigma^2_i)\ \text{i.i.d.},

with the factor returns drawn from a chosen covariance and the idiosyncratic
noise independent across assets — exactly the assumptions the factor risk model
inverts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantica.core.types import FloatArray

__all__ = [
    "DEFAULT_FACTOR_NAMES",
    "SyntheticFactorData",
    "generate_factor_data",
]

#: The Fama--French--Carhart observable factors, in the conventional order.
DEFAULT_FACTOR_NAMES = ("MKT", "SMB", "HML", "MOM")

# Plausible monthly factor volatilities (decimal, ~ US equity history), used when
# the caller does not supply their own.
_DEFAULT_FACTOR_VOLS = (0.045, 0.030, 0.030, 0.045)


@dataclass(frozen=True)
class SyntheticFactorData:
    """A simulated panel of asset and factor returns with its ground truth.

    Attributes
    ----------
    asset_returns : ndarray, shape (T, n_assets)
        Excess returns of the assets.
    factor_returns : ndarray, shape (T, k)
        The factor excess returns.
    true_betas : ndarray, shape (n_assets, k)
        The planted factor loadings ``B`` — what a fit must recover.
    true_alphas : ndarray, shape (n_assets,)
        The planted intercepts.
    true_specific_var : ndarray, shape (n_assets,)
        The planted idiosyncratic variances (the diagonal of ``D``).
    asset_names, factor_names : tuple of str
        Column labels.
    """

    asset_returns: FloatArray
    factor_returns: FloatArray
    true_betas: FloatArray
    true_alphas: FloatArray
    true_specific_var: FloatArray
    asset_names: tuple[str, ...]
    factor_names: tuple[str, ...]


def generate_factor_data(
    n_periods: int,
    betas: FloatArray,
    rng: np.random.Generator,
    *,
    factor_vols: FloatArray | None = None,
    factor_correlation: FloatArray | None = None,
    specific_vols: FloatArray | float = 0.05,
    alphas: FloatArray | float = 0.0,
    factor_names: tuple[str, ...] = DEFAULT_FACTOR_NAMES,
) -> SyntheticFactorData:
    r"""Draw a return panel from the known linear factor model.

    Parameters
    ----------
    n_periods : int
        Number of time periods ``T`` (must be positive).
    betas : ndarray, shape (n_assets, k)
        The planted loadings; ``k`` must match ``len(factor_names)``.
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    factor_vols : ndarray, shape (k,), optional
        Per-factor volatilities; defaults to plausible monthly equity values.
    factor_correlation : ndarray, shape (k, k), optional
        Factor correlation matrix; defaults to the identity (uncorrelated
        factors, so the recovered factor covariance is diagonal by construction).
    specific_vols : ndarray (n_assets,) or float, optional
        Idiosyncratic volatilities (default 0.05); a scalar broadcasts.
    alphas : ndarray (n_assets,) or float, optional
        Intercepts (default 0); a scalar broadcasts.
    factor_names : tuple of str, optional
        Factor labels (default Fama--French--Carhart).
    """
    if n_periods < 1:
        raise ValueError(f"n_periods must be at least 1, got {n_periods}")
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError(f"betas must be 2-D (n_assets, k), got shape {b.shape}")
    n_assets, k = b.shape
    if k != len(factor_names):
        raise ValueError(f"betas has {k} factors but {len(factor_names)} names were given")

    vols = (
        np.asarray(_DEFAULT_FACTOR_VOLS[:k], dtype=np.float64)
        if factor_vols is None
        else np.asarray(factor_vols, dtype=np.float64)
    )
    if vols.shape != (k,):
        raise ValueError(f"factor_vols must have shape ({k},), got {vols.shape}")
    corr = np.eye(k) if factor_correlation is None else np.asarray(factor_correlation, np.float64)
    if corr.shape != (k, k):
        raise ValueError(f"factor_correlation must be ({k}, {k}), got {corr.shape}")

    factor_cov = np.outer(vols, vols) * corr
    factor_returns = rng.multivariate_normal(np.zeros(k), factor_cov, size=n_periods)

    spec_vols = np.broadcast_to(np.asarray(specific_vols, dtype=np.float64), (n_assets,)).copy()
    alpha_vec = np.broadcast_to(np.asarray(alphas, dtype=np.float64), (n_assets,)).copy()
    noise = rng.normal(0.0, 1.0, size=(n_periods, n_assets)) * spec_vols

    asset_returns = alpha_vec + factor_returns @ b.T + noise
    return SyntheticFactorData(
        asset_returns=asset_returns,
        factor_returns=factor_returns,
        true_betas=b,
        true_alphas=alpha_vec,
        true_specific_var=spec_vols**2,
        asset_names=tuple(f"A{i + 1}" for i in range(n_assets)),
        factor_names=tuple(factor_names),
    )
