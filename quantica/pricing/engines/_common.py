"""Shared helpers for pricing engines.

Extracted once a *second* engine needed the same scalar parameter bundle
(CLAUDE.md §2, "no premature abstraction"): the analytic and binomial engines
both read the same fields off an instrument/process pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from quantica.pricing.instruments import EuropeanOption
    from quantica.pricing.processes import BlackScholesProcess


class BSParams(NamedTuple):
    r"""The scalar Black--Scholes inputs, in the usual notation.

    ``sign`` is the payoff sign :math:`\omega` (``+1`` call, ``-1`` put).
    """

    spot: float  # S
    strike: float  # K
    rate: float  # r
    div: float  # q
    vol: float  # sigma
    expiry: float  # T
    sign: int  # omega


def unpack(instrument: EuropeanOption, process: BlackScholesProcess) -> BSParams:
    """Bundle the scalar model parameters from an instrument and process."""
    return BSParams(
        spot=process.spot,
        strike=instrument.strike,
        rate=process.rate,
        div=process.div,
        vol=process.vol,
        expiry=instrument.expiry,
        sign=instrument.option_type.sign,
    )
