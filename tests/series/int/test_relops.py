from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import np_ndarray_dt


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


def test_relops_py_sequence(left: pd.Series[int]) -> None:
    values = [1.5, 2.5, 3.5]

    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array(left: pd.Series[int]) -> None:
    array = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    check(assert_type(left < array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= array, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.le(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(array), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_invalid_numpy_array(left: pd.Series[int]) -> None:
    other: np_ndarray_dt = np.array(["2022-01-01"] * 3, dtype="datetime64[ns]")

    if TYPE_CHECKING_INVALID_USAGE:
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
