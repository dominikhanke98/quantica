"""Tests for the shared enums and type aliases."""

from __future__ import annotations

from quantica.core.types import AveragingType, BarrierType, ExerciseStyle, OptionType


def test_option_type_sign() -> None:
    assert OptionType.CALL.sign == 1
    assert OptionType.PUT.sign == -1


def test_option_type_str() -> None:
    assert str(OptionType.CALL) == "call"
    assert str(OptionType.PUT) == "put"


def test_option_type_members_are_distinct() -> None:
    assert OptionType.CALL is not OptionType.PUT
    assert {OptionType.CALL, OptionType.PUT} == set(OptionType)


def test_exercise_style_str_and_members() -> None:
    assert str(ExerciseStyle.EUROPEAN) == "european"
    assert str(ExerciseStyle.AMERICAN) == "american"
    assert ExerciseStyle.EUROPEAN is not ExerciseStyle.AMERICAN


def test_averaging_type_str() -> None:
    assert str(AveragingType.ARITHMETIC) == "arithmetic"
    assert str(AveragingType.GEOMETRIC) == "geometric"


def test_barrier_type_direction_and_knock_and_str() -> None:
    assert BarrierType.UP_AND_OUT.is_up and not BarrierType.DOWN_AND_IN.is_up
    assert BarrierType.DOWN_AND_IN.is_knock_in and not BarrierType.UP_AND_OUT.is_knock_in
    assert str(BarrierType.DOWN_AND_OUT) == "down-and-out"
