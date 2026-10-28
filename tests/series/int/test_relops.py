from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> pd.Series[int]:
    """Left operand"""
    lo = pd.Series([1, 2, 3])
    return check(assert_type(lo, "pd.Series[int]"), pd.Series, np.integer)


def test_relops_py_scalar(left: pd.Series[int]) -> None:
    """Test pd.Series[int] <comparison> py scalar"""
    # GH 175
    i = 10

    check(assert_type(left < i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= i, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(i), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(i), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(i), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(i), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_mixed_numeric(left: pd.Series[int]) -> None:
    """Test pd.Series[int] <comparison> float scalar"""
    # GH 1903
    f = 3.14

    check(assert_type(left < f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= f, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(f < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(f), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence(left: pd.Series[int]) -> None:
    values = [1.5, 2.5, 3.5]

    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array(left: pd.Series[int]) -> None:
    array = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    check(assert_type(left < array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= array, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.__lt__(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__le__(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__gt__(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__ge__(array), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(array), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_numpy_array(left: pd.Series[int]) -> None:
    other = np.array([np.datetime64("2022-01-01", "ns")] * 3, dtype=np.datetime64)

    if TYPE_CHECKING_INVALID_USAGE:
        left.__lt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__le__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__gt__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.__ge__(other)  # type: ignore[operator] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]

        left.lt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.le(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.gt(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]
        left.ge(other)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue,reportArgumentType] # pyrefly: ignore[no-matching-overload] # ty: ignore[no-matching-overload]


def test_relops_pd_index(left: pd.Series[int]) -> None:
    index = pd.Index(left)
    f = pd.Index([1.5, 2.5, 3.5])
    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= index, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= f, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(index), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(f), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_series(left: pd.Series[int]) -> None:
    other = left.copy()
    f = pd.Series([1.5, 2.5, 3.5])
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= f, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(f < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(f), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(f), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(f.lt(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f.le(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f.gt(left), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f.ge(left), "pd.Series[bool]"), pd.Series, np.bool_)
