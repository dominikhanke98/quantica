r"""A seeded synthetic credit portfolio — the reproducible test bed.

Public loan-level default data with a known ground truth does not exist, so the
validation battery is exercised on a **synthetic portfolio whose true PDs are
known by construction** — which is exactly what the meta-validation ("validate
the validators") requires: only with a known truth can the *size* and *power* of
the calibration tests be measured.

The generative model is a latent log-odds with two deliberate wrinkles:

* an **interaction** (leverage times behavioural) and a **convexity**
  (leverage squared), so a linear-logit champion is mildly mis-specified while a
  gradient-boosting challenger has genuine signal to find — reproducing the
  classic champion/challenger situation on demand;
* a ``leverage_shift`` knob that translates the leverage distribution of a
  monitoring sample, giving the stability metrics (PSI/CSI) a controlled drift
  to detect.

Everything is driven by an injected, seeded ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantica.core.types import FloatArray

__all__ = ["CreditSample", "generate_credit_portfolio"]

#: Obligor characteristics (all standardised to ~N(0,1) in the development population).
FEATURE_NAMES = ("leverage", "profitability", "liquidity", "size", "behavioural")

# True log-odds coefficients. Signs follow credit intuition: higher leverage and a
# worse behavioural score raise PD; profitability, liquidity and size lower it.
_INTERCEPT = -4.6
_BETA = (0.9, -0.7, -0.5, -0.3, 0.6)  # aligned with FEATURE_NAMES
_BETA_INTERACTION = 1.0  # leverage x behavioural
_BETA_CONVEXITY = 0.55  # leverage^2 (risk accelerates in the tail)


@dataclass(frozen=True)
class CreditSample:
    """A simulated loan book with its ground truth.

    Attributes
    ----------
    features : ndarray, shape (n, 5)
        Obligor characteristics, columns ordered as ``feature_names``.
    defaults : ndarray, shape (n,)
        Realised 0/1 default indicators, drawn Bernoulli(``true_pd``).
    true_pd : ndarray, shape (n,)
        The generative per-obligor PDs — the truth a validator can be scored
        against (never available in production; the point of synthetic data).
    feature_names : tuple of str
        Column names for ``features``.
    """

    features: FloatArray
    defaults: FloatArray
    true_pd: FloatArray
    feature_names: tuple[str, ...]


def generate_credit_portfolio(
    n_obligors: int,
    rng: np.random.Generator,
    *,
    leverage_shift: float = 0.0,
) -> CreditSample:
    """Draw a synthetic loan book from the known generative PD model.

    Parameters
    ----------
    n_obligors : int
        Number of obligors (must be positive).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    leverage_shift : float, optional
        Additive shift of the leverage distribution — zero for a development
        sample; non-zero to simulate the population drift of a monitoring sample
        (raises portfolio risk and moves the PSI).
    """
    if n_obligors < 1:
        raise ValueError(f"n_obligors must be at least 1, got {n_obligors}")
    x = rng.standard_normal((n_obligors, len(FEATURE_NAMES)))
    x[:, 0] += leverage_shift

    log_odds = _INTERCEPT + x @ np.asarray(_BETA)
    log_odds += _BETA_INTERACTION * x[:, 0] * x[:, 4] + _BETA_CONVEXITY * x[:, 0] ** 2
    true_pd = 1.0 / (1.0 + np.exp(-log_odds))
    defaults = (rng.random(n_obligors) < true_pd).astype(np.float64)
    return CreditSample(
        features=x,
        defaults=defaults,
        true_pd=true_pd,
        feature_names=FEATURE_NAMES,
    )
