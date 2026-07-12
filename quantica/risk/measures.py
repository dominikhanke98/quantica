r"""Value-at-Risk and Expected Shortfall — definitions and closed forms.

Conventions
-----------
Everything here works in **loss** units: a loss is :math:`L = -\text{P\&L}`, so a
positive number is money lost. For a confidence level :math:`\alpha` (e.g. 0.99):

* **Value-at-Risk** :math:`\mathrm{VaR}_\alpha` is the :math:`\alpha`-quantile of
  the loss distribution — the loss that is exceeded with probability
  :math:`1-\alpha`.
* **Expected Shortfall** :math:`\mathrm{ES}_\alpha = \mathbb E[L \mid L \ge
  \mathrm{VaR}_\alpha]` is the average loss *in that tail* — a coherent risk
  measure, unlike VaR.

Two constructors are provided: an :func:`empirical_var_es` estimator from a loss
sample (used by historical / filtered-historical / Monte-Carlo engines) and the
Gaussian closed form :func:`normal_var_es` (the parametric engine, and the analytic
anchor the others are validated against).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from quantica.core.types import FloatArray

__all__ = ["RiskEstimate", "empirical_var_es", "normal_var_es"]


@dataclass(frozen=True)
class RiskEstimate:
    r"""A VaR/ES estimate at one confidence level, in loss (money) units.

    Attributes
    ----------
    var : float
        Value-at-Risk :math:`\mathrm{VaR}_\alpha` (a positive loss).
    es : float
        Expected Shortfall :math:`\mathrm{ES}_\alpha \ge \mathrm{VaR}_\alpha`.
    level : float
        The confidence level :math:`\alpha \in (0, 1)`.
    method : str
        Which engine produced it (for reporting/tables).
    """

    var: float
    es: float
    level: float
    method: str = ""


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def normal_var_es(
    mean: float, sigma: float, level: float, *, method: str = "parametric-normal"
) -> RiskEstimate:
    r"""Gaussian closed-form VaR and ES for P\&L :math:`\sim \mathcal N(\mu, \sigma^2)`.

    With losses :math:`L = -\text{P\&L} \sim \mathcal N(-\mu, \sigma^2)` and
    :math:`z_\alpha = \Phi^{-1}(\alpha)`,

    .. math::

        \mathrm{VaR}_\alpha = -\mu + \sigma z_\alpha, \qquad
        \mathrm{ES}_\alpha = -\mu + \sigma\,\frac{\phi(z_\alpha)}{1 - \alpha}.

    Parameters
    ----------
    mean : float
        Mean of the P\&L (drift), in money units.
    sigma : float
        Standard deviation of the P\&L. Must be non-negative.
    level : float
        Confidence level :math:`\alpha`.
    """
    _check_level(level)
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    z = float(norm.ppf(level))
    var = -mean + sigma * z
    es = -mean + sigma * float(norm.pdf(z)) / (1.0 - level)
    return RiskEstimate(var=var, es=es, level=level, method=method)


def empirical_var_es(
    losses: FloatArray, level: float, *, method: str = "historical"
) -> RiskEstimate:
    r"""Empirical VaR and ES from a sample of losses.

    ``VaR`` is the linear-interpolated :math:`\alpha`-quantile of ``losses``. ``ES``
    uses the Rockafellar--Uryasev tail-mean identity

    .. math::

        \mathrm{ES}_\alpha = \mathrm{VaR}_\alpha
            + \frac{1}{1-\alpha}\,\mathbb E\big[(L - \mathrm{VaR}_\alpha)^+\big],

    which is numerically stable (no reliance on how many points land exactly in the
    tail) and coincides with the average of the worst :math:`(1-\alpha)` fraction of
    losses for a continuous distribution.

    Parameters
    ----------
    losses : ndarray
        Sample of losses (positive = loss).
    level : float
        Confidence level :math:`\alpha`.
    """
    _check_level(level)
    sample = np.asarray(losses, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("losses must be a non-empty 1-D array")
    var = float(np.quantile(sample, level))
    es = var + float(np.mean(np.maximum(sample - var, 0.0))) / (1.0 - level)
    return RiskEstimate(var=var, es=es, level=level, method=method)
