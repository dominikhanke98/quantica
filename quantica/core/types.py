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
