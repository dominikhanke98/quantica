"""Shared enums and type aliases used across the package.

Notes
-----
The float-array alias keeps signatures readable while remaining precise enough
for ``mypy --strict`` and the numpy plugin. Pricers operate on
double-precision arrays throughout.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import numpy.typing as npt

#: Double-precision float array, the working type for spot grids and payoffs.
FloatArray = npt.NDArray[np.float64]

#: Anything that can be broadcast to a :data:`FloatArray` (scalars or arrays).
FloatLike = float | FloatArray


class OptionType(Enum):
    """The two vanilla option flavours.

    The integer value is the payoff *sign* applied to ``spot - strike``:
    ``+1`` for a call, ``-1`` for a put. This makes the payoff and the
    Black--Scholes closed form expressible without branching, e.g.
    ``max(sign * (spot - strike), 0)``.
    """

    CALL = 1
    PUT = -1

    @property
    def sign(self) -> int:
        """Payoff sign: ``+1`` for a call, ``-1`` for a put."""
        return self.value

    def __str__(self) -> str:
        return self.name.lower()


class ExerciseStyle(Enum):
    """When the holder may exercise.

    ``EUROPEAN`` — only at expiry (closed-form Black--Scholes applies).
    ``AMERICAN`` — at any time up to expiry (no closed form; priced by lattice
    or PDE with an early-exercise/free-boundary condition).
    """

    EUROPEAN = "european"
    AMERICAN = "american"

    def __str__(self) -> str:
        return self.value


class AveragingType(Enum):
    """How an Asian option averages the underlying over the monitoring dates.

    ``ARITHMETIC`` — the arithmetic mean (no closed form under GBM).
    ``GEOMETRIC`` — the geometric mean (a lognormal, hence a closed form; the
    natural control variate for the arithmetic average).
    """

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"

    def __str__(self) -> str:
        return self.value


class BarrierType(Enum):
    """The four single-barrier knock styles.

    Named ``<direction>_AND_<knock>``: ``direction`` is where the barrier sits
    relative to spot (``UP`` above, ``DOWN`` below); ``knock`` is whether hitting
    it activates (``IN``) or extinguishes (``OUT``) the option.
    """

    UP_AND_OUT = "up-and-out"
    UP_AND_IN = "up-and-in"
    DOWN_AND_OUT = "down-and-out"
    DOWN_AND_IN = "down-and-in"

    @property
    def is_up(self) -> bool:
        """True if the barrier is above the spot (an *up* barrier)."""
        return self in (BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN)

    @property
    def is_knock_in(self) -> bool:
        """True if hitting the barrier *activates* the option (a knock-*in*)."""
        return self in (BarrierType.UP_AND_IN, BarrierType.DOWN_AND_IN)

    def __str__(self) -> str:
        return self.value
