"""Multi-factor risk models — the shared foundation for risk and portfolio work.

A factor model compresses the asset covariance into loadings, a factor covariance,
and specific risks (:math:`\\Sigma = B F B^\\top + D`). It underpins both
market-risk decomposition and portfolio construction, so it lives at the top level
rather than inside either consumer.

This is a **top-level** package (not under ``risk`` or ``portfolio``) precisely
because both pillars consume it. Estimation leans on established libraries
(statsmodels OLS for the loadings, ``numpy.cov`` for the factor covariance); the
package's own value is the risk-model assembly, decomposition, and the
out-of-sample estimator-comparison layer (stage 2).
"""

from __future__ import annotations

from quantica.factor.data import (
    DEFAULT_FACTOR_NAMES,
    SyntheticFactorData,
    generate_factor_data,
)
from quantica.factor.estimators import (
    CovarianceEstimator,
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    condition_number,
    min_variance_weights,
)
from quantica.factor.evaluation import (
    BiasStats,
    EstimatorComparison,
    WalkForwardWindow,
    compare_estimators,
    frobenius_error,
    min_variance_true_loss,
    walk_forward_windows,
)
from quantica.factor.exposures import FactorExposures, estimate_exposures
from quantica.factor.model import (
    AssetVarianceDecomposition,
    FactorRiskModel,
    LinearFactorModel,
    PortfolioRiskDecomposition,
)
from quantica.factor.statistical import (
    StatisticalFactorCovariance,
    StatisticalFactorModel,
    marchenko_pastur_edges,
    marchenko_pastur_rank,
    scree_elbow_rank,
    subspace_similarity,
    variance_explained_rank,
)

__all__ = [
    "DEFAULT_FACTOR_NAMES",
    "AssetVarianceDecomposition",
    "BiasStats",
    "CovarianceEstimator",
    "EstimatorComparison",
    "FactorCovariance",
    "FactorExposures",
    "FactorRiskModel",
    "LedoitWolfCovariance",
    "LinearFactorModel",
    "PortfolioRiskDecomposition",
    "SampleCovariance",
    "StatisticalFactorCovariance",
    "StatisticalFactorModel",
    "SyntheticFactorData",
    "WalkForwardWindow",
    "compare_estimators",
    "condition_number",
    "estimate_exposures",
    "frobenius_error",
    "generate_factor_data",
    "marchenko_pastur_edges",
    "marchenko_pastur_rank",
    "min_variance_true_loss",
    "min_variance_weights",
    "scree_elbow_rank",
    "subspace_similarity",
    "variance_explained_rank",
    "walk_forward_windows",
]
