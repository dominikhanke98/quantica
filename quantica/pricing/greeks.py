"""The Greeks — first-order price sensitivities.

A small, engine-agnostic value object so any engine that computes sensitivities
returns them in one shape. Conventions (all w.r.t. the *raw* parameter, so they
line up one-to-one with a bump-and-reval of that parameter):

============  ===================================  ================
Greek         Definition                           Bumped parameter
============  ===================================  ================
``delta``     :math:`\\partial V / \\partial S`        spot
``gamma``     :math:`\\partial^2 V / \\partial S^2`     spot (2nd order)
``vega``      :math:`\\partial V / \\partial \\sigma`    volatility
``theta``     :math:`\\partial V / \\partial t`         calendar time
``rho``       :math:`\\partial V / \\partial r`         risk-free rate
============  ===================================  ================

``vega`` and ``rho`` are per unit (not per 1%), and ``theta`` is per year (not
per day) — the natural units for validating against a finite difference of the
corresponding parameter.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Greeks:
    """First-order sensitivities of an option price. See module docstring."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
