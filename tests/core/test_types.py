"""Tests for the shared enums and type aliases."""

from __future__ import annotations

from quantica.core.types import OptionType


def test_option_type_sign() -> None:
    assert OptionType.CALL.sign == 1
    assert OptionType.PUT.sign == -1


def test_option_type_str() -> None:
    assert str(OptionType.CALL) == "call"
    assert str(OptionType.PUT) == "put"


def test_option_type_members_are_distinct() -> None:
    assert OptionType.CALL is not OptionType.PUT
    assert {OptionType.CALL, OptionType.PUT} == set(OptionType)
