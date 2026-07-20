r"""Statistical (PCA) factor model — the factors are estimated, not observed.

The observable model (:class:`~quantica.factor.model.FactorRiskModel`) takes the factors
as *given* (Fama--French market/size/value/momentum) and regresses on them. The
**statistical** model instead *discovers* the factors from the asset returns themselves:
the principal components of the return correlation matrix are the factors, and the
eigenvector-scaled loadings are the exposures. It completes the observable-vs-statistical
pair, sharing the :math:`\Sigma = B F B^\top + D` assembly and decomposition through the
common :class:`~quantica.factor.model.LinearFactorModel` base, so a PCA model plugs into
the same risk decomposition and the same out-of-sample estimator comparison.

Construction (correlation-PCA, so the RMT cutoff below is clean)
----------------------------------------------------------------
Standardise the returns to unit variance, :math:`Z = (R-\bar R)/s`, and eigendecompose
their correlation matrix :math:`C = Z^\top Z/(T-1) = V\Lambda V^\top` (equivalently, SVD
of :math:`Z`). Keeping the top ``k`` components, the *correlation-space* loadings are
:math:`L = V_k\Lambda_k^{1/2}` and the model in return space is

.. math::

    B = \operatorname{diag}(s)\,L,\qquad F = I_k,\qquad
    D = \operatorname{diag}\!\big(s^2\,(1 - \textstyle\sum_j L_{\cdot j}^2)\big),

so :math:`\Sigma = B B^\top + D`. The factors are orthonormal (variance folded into the
loadings), hence :math:`F = I_k`. Because the kept plus dropped communality is exactly one
(unit correlation diagonal), the reconstruction **preserves the sample variances exactly**
on the diagonal and only approximates the off-diagonal cross-correlations — a clean anchor.

Component selection is a modelling decision, not a hard-coded ``k``
------------------------------------------------------------------
Three principled rules are provided, the last being the differentiator:

* :func:`variance_explained_rank` — the smallest ``k`` reaching a cumulative
  variance-explained threshold.
* :func:`scree_elbow_rank` — the geometric elbow of the scree curve (distance to the chord).
* :func:`marchenko_pastur_rank` — the **random-matrix-theory** cutoff: eigenvalues above the
  Marchenko--Pastur upper edge :math:`\lambda_+ = \sigma^2(1+\sqrt{n/T})^2` are
  statistically *real*; those inside the bulk are indistinguishable from noise. This says
  "here are the factors that are signal, not sampling artefact", and — with the optional
  bulk-variance fit — is the default.

References
----------
Marchenko, V. A. & Pastur, L. A. (1967). "Distribution of eigenvalues for some sets of
random matrices", *Math. USSR-Sbornik*.
Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). "Noise dressing of financial
correlation matrices", *Physical Review Letters* 83, 1467.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantica.core.types import FloatArray
from quantica.factor.model import LinearFactorModel

__all__ = [
    "StatisticalFactorCovariance",
    "StatisticalFactorModel",
    "marchenko_pastur_edges",
    "marchenko_pastur_rank",
    "scree_elbow_rank",
    "subspace_similarity",
    "variance_explained_rank",
]


# --------------------------------------------------------------------------- #
# Component-selection rules (each maps a spectrum to a factor count k)
# --------------------------------------------------------------------------- #


def variance_explained_rank(eigenvalues: FloatArray, threshold: float = 0.9) -> int:
    """Smallest ``k`` whose top eigenvalues explain at least ``threshold`` of the variance.

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues (any order; sorted descending internally).
    threshold : float, optional
        Target cumulative variance-explained fraction in ``(0, 1]`` (default 0.9).

    Returns
    -------
    int
        The number of leading components reaching the threshold (at least 1).
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    vals = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    total = vals.sum()
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(vals) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def scree_elbow_rank(eigenvalues: FloatArray) -> int:
    """The scree-plot elbow: the component furthest from the first-to-last chord.

    Draws the straight line from the first to the last (sorted, descending) eigenvalue and
    returns the index whose eigenvalue lies furthest *below* it — the classic geometric
    elbow (a Kneedle-style rule). Robust to the overall scale of the spectrum.

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues (any order; sorted descending internally).

    Returns
    -------
    int
        The elbow component count (at least 1).
    """
    vals = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    n = vals.size
    if n <= 2:
        return int(n)
    x = np.arange(n, dtype=np.float64)
    # Distance from each (x, eigenvalue) point to the chord through the endpoints.
    x0, y0, x1, y1 = x[0], vals[0], x[-1], vals[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = np.hypot(dx, dy)
    distances = np.abs(dy * (x - x0) - dx * (vals - y0)) / norm
    # The furthest point below the chord is the first eigenvalue *after* the drop, so its
    # 0-based index equals the count of large components to keep (e.g. index 3 -> keep 3).
    return max(1, int(np.argmax(distances)))


def marchenko_pastur_edges(n_assets: int, n_obs: int, variance: float = 1.0) -> tuple[float, float]:
    r"""The Marchenko--Pastur bulk edges :math:`\lambda_\pm = \sigma^2(1\pm\sqrt{n/T})^2`.

    For a correlation matrix of ``n_assets`` pure-noise series over ``n_obs`` observations,
    the sample eigenvalues fill the interval :math:`[\lambda_-, \lambda_+]`; anything above
    :math:`\lambda_+` is signal, not noise.

    Parameters
    ----------
    n_assets : int
        Number of assets ``n``.
    n_obs : int
        Number of observations ``T``.
    variance : float, optional
        The bulk (noise) variance :math:`\sigma^2` (default 1 for a correlation matrix).

    Returns
    -------
    tuple of float
        ``(lambda_minus, lambda_plus)``.
    """
    if n_assets < 1 or n_obs < 1:
        raise ValueError("n_assets and n_obs must be positive")
    q = np.sqrt(n_assets / n_obs)
    lam_minus = variance * (1.0 - q) ** 2
    lam_plus = variance * (1.0 + q) ** 2
    return float(lam_minus), float(lam_plus)


def marchenko_pastur_rank(
    eigenvalues: FloatArray,
    n_assets: int,
    n_obs: int,
    *,
    adjust_variance: bool = False,
) -> int:
    r"""Count the eigenvalues above the Marchenko--Pastur upper edge (the RMT signal count).

    Eigenvalues exceeding :math:`\lambda_+` cannot be explained by sampling noise, so their
    count is the number of statistically real factors. With ``adjust_variance`` the bulk
    (noise) variance is fitted iteratively — the leading "signal" eigenvalues carry variance
    out of the bulk, so the effective :math:`\sigma^2` is below one, and re-estimating it
    from the remaining trace sharpens the edge (Laloux et al. 1999).

    Parameters
    ----------
    eigenvalues : ndarray, shape (n,)
        Correlation-matrix eigenvalues (any order; sorted descending internally).
    n_assets, n_obs : int
        Universe size ``n`` and observation count ``T``.
    adjust_variance : bool, optional
        Refit the bulk variance once after removing the dominant modes (default ``True``);
        if ``False`` use :math:`\sigma^2 = 1`.

    Returns
    -------
    int
        The number of eigenvalues above the (possibly refitted) upper edge — **may be 0**
        for pure noise, which is the correct, informative answer.

    Notes
    -----
    With a correlation matrix a dominant "market" mode carries a large share of the trace
    :math:`n`, which depletes the bulk and leaves :math:`\sigma^2 = 1` *over*-estimating the
    noise edge (so the weaker real factors, sitting just above one, hide inside the bulk).
    The refit removes the modes found at :math:`\sigma^2 = 1` and re-estimates the bulk
    variance from the remaining trace, :math:`\sigma^2 = (n - \sum_{\text{top }k}\lambda) /
    (n - k)`, then recounts — a single, stable pass (iterating to a fixed point instead
    spirals on finite-size stragglers just above the edge).
    """
    vals = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    n = vals.size
    _, lam_plus = marchenko_pastur_edges(n_assets, n_obs)
    k = int(np.sum(vals > lam_plus))
    if not adjust_variance or k == 0 or k >= n:
        return k
    variance = float((n - vals[:k].sum()) / (n - k))  # bulk variance after removing signal
    _, lam_plus = marchenko_pastur_edges(n_assets, n_obs, variance)
    return int(np.sum(vals > lam_plus))


def subspace_similarity(loadings: FloatArray, reference: FloatArray) -> float:
    r"""Cosine of the largest principal angle between two loading subspaces (1 = identical).

    Individual principal components are not identified (rotations/sign flips within the
    factor space are free), but the *span* is. This compares the column space of two loading
    matrices via the principal angles (``scipy.linalg.subspace_angles``): the cosine of the
    largest angle is ``1`` when one span contains the other and falls toward ``0`` as they
    diverge — the right way to check "PCA recovered the true factor space".

    Parameters
    ----------
    loadings : ndarray, shape (n, k)
        Estimated loadings whose column span is tested.
    reference : ndarray, shape (n, m)
        Reference loadings (e.g. the true planted betas).

    Returns
    -------
    float
        ``cos`` of the largest principal angle between the two column spans.
    """
    from scipy.linalg import subspace_angles  # lazy: scipy.linalg import

    a = np.asarray(loadings, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    angles = subspace_angles(a, b)
    if angles.size == 0:
        return 1.0
    return float(np.cos(np.max(angles)))


# --------------------------------------------------------------------------- #
# The statistical factor model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StatisticalFactorModel(LinearFactorModel):
    r"""A fitted **statistical** (PCA) factor model :math:`\Sigma = B B^\top + D`.

    Build with :meth:`fit`. Extends :class:`~quantica.factor.model.LinearFactorModel` with
    the full eigenvalue spectrum and the fitting metadata; the covariance and decomposition
    methods are inherited. The factors are the principal components of the return
    correlation matrix, so :attr:`factor_cov` is the identity (orthonormal factors).

    Attributes
    ----------
    eigenvalues : ndarray, shape (n,)
        The full correlation-matrix spectrum, descending — the scree data.
    n_obs : int
        Number of observations the model was fitted on.
    selection : str
        The component-selection rule used (``"marchenko_pastur"`` / ``"variance"`` /
        ``"scree"`` / ``"fixed"``).
    """

    eigenvalues: FloatArray
    n_obs: int
    selection: str

    @property
    def n_factors(self) -> int:
        """The number of retained statistical factors ``k``."""
        return int(self.betas.shape[1])

    @property
    def explained_variance_ratio(self) -> FloatArray:
        """Fraction of total variance carried by each component (descending, sums to 1)."""
        total = float(self.eigenvalues.sum())
        return np.asarray(self.eigenvalues / total, dtype=np.float64)

    @property
    def cumulative_variance_ratio(self) -> FloatArray:
        """Cumulative :attr:`explained_variance_ratio` (the variance-explained curve)."""
        return np.asarray(np.cumsum(self.explained_variance_ratio), dtype=np.float64)

    @classmethod
    def fit(
        cls,
        asset_returns: FloatArray,
        *,
        n_components: int | None = None,
        selection: str = "marchenko_pastur",
        variance_threshold: float = 0.9,
        adjust_variance: bool = False,
        asset_names: tuple[str, ...] | None = None,
    ) -> StatisticalFactorModel:
        r"""Fit the PCA factor model to a return panel.

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n)
            The asset return panel.
        n_components : int, optional
            Force a fixed number of factors, bypassing ``selection``.
        selection : {"marchenko_pastur", "variance", "scree"}, optional
            The component-selection rule when ``n_components`` is ``None`` (default the RMT
            Marchenko--Pastur cutoff).
        variance_threshold : float, optional
            Threshold for the ``"variance"`` rule (default 0.9).
        adjust_variance : bool, optional
            Refit the bulk variance in the Marchenko--Pastur rule (default ``False`` — the
            plain :math:`\sigma^2=1` textbook cutoff; see :func:`marchenko_pastur_rank`).
        asset_names : tuple of str, optional
            Asset labels; default ``A1..An``.

        Returns
        -------
        StatisticalFactorModel
            The fitted model, retaining at least one factor.

        Raises
        ------
        ValueError
            If ``asset_returns`` is not 2-D, has fewer than two observations, or
            ``n_components`` is out of range.
        """
        r = np.asarray(asset_returns, dtype=np.float64)
        if r.ndim != 2:
            raise ValueError(f"asset_returns must be 2-D (T, n), got shape {r.shape}")
        n_obs, n_assets = r.shape
        if n_obs < 2:
            raise ValueError(f"need at least 2 observations, got {n_obs}")
        asset_names = asset_names or tuple(f"A{i + 1}" for i in range(n_assets))
        if len(asset_names) != n_assets:
            raise ValueError(f"{len(asset_names)} asset_names for {n_assets} assets")

        std = r.std(axis=0, ddof=1)
        standardized = (r - r.mean(axis=0)) / std
        correlation = np.atleast_2d(np.cov(standardized, rowvar=False, ddof=1))
        raw_vals, raw_vecs = np.linalg.eigh(correlation)  # ascending, symmetric
        order = np.argsort(raw_vals)[::-1]
        eigenvalues = np.clip(raw_vals[order], 0.0, None)
        eigenvectors = raw_vecs[:, order]

        if n_components is not None:
            if not 1 <= n_components <= n_assets:
                raise ValueError(f"n_components must be in [1, {n_assets}], got {n_components}")
            k, rule = n_components, "fixed"
        elif selection == "variance":
            k, rule = variance_explained_rank(eigenvalues, variance_threshold), "variance"
        elif selection == "scree":
            k, rule = scree_elbow_rank(eigenvalues), "scree"
        elif selection == "marchenko_pastur":
            k = marchenko_pastur_rank(eigenvalues, n_assets, n_obs, adjust_variance=adjust_variance)
            rule = "marchenko_pastur"
        else:
            raise ValueError(f"unknown selection rule {selection!r}")
        k = max(1, min(k, n_assets))

        top_vecs = eigenvectors[:, :k]
        top_vals = eigenvalues[:k]
        corr_loadings = top_vecs * np.sqrt(top_vals)  # L = V_k Lambda_k^{1/2}
        betas = std[:, None] * corr_loadings  # B = diag(s) L
        communality = np.sum(corr_loadings * corr_loadings, axis=1)
        specific_var = std * std * np.clip(1.0 - communality, 0.0, None)

        return cls(
            asset_names=asset_names,
            factor_names=tuple(f"PC{j + 1}" for j in range(k)),
            betas=np.asarray(betas, dtype=np.float64),
            factor_cov=np.eye(k, dtype=np.float64),
            specific_var=np.asarray(specific_var, dtype=np.float64),
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            n_obs=n_obs,
            selection=rule,
        )


class StatisticalFactorCovariance:
    """A PCA-factor :class:`~quantica.factor.estimators.CovarianceEstimator`.

    Estimates the covariance as the statistical-factor reconstruction
    :math:`\\Sigma = B B^\\top + D`. Unlike
    :class:`~quantica.factor.estimators.FactorCovariance`, it needs **no** observable factor
    returns — it discovers the factors from the asset returns — so it slots into the stage-2
    out-of-sample comparison as the statistical counterpart of the observable factor model.
    """

    name = "statistical-factor"

    def __init__(
        self,
        *,
        selection: str = "marchenko_pastur",
        n_components: int | None = None,
        variance_threshold: float = 0.9,
    ) -> None:
        self.selection = selection
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def estimate(
        self, asset_returns: FloatArray, factor_returns: FloatArray | None = None
    ) -> FloatArray:
        """Return the PCA-factor covariance of ``asset_returns``.

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n)
            The asset return panel.
        factor_returns : ndarray, optional
            Unused — statistical factors are estimated from the asset returns; accepted for
            interface compatibility with :class:`~quantica.factor.estimators.CovarianceEstimator`.

        Returns
        -------
        ndarray, shape (n, n)
            The reconstructed covariance matrix.
        """
        model = StatisticalFactorModel.fit(
            asset_returns,
            n_components=self.n_components,
            selection=self.selection,
            variance_threshold=self.variance_threshold,
        )
        return model.covariance()
