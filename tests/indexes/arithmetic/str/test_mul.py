from datetime import (
    datetime,
    timedelta,
)
from typing import Any

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
import pytest
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)


@pytest.fixture
def left() -> "pd.Index[str]":
    """left operand"""
    lo = pd.Index(["1", "2", "3"])
    return check(assert_type(lo, "pd.Index[str]"), pd.Index, str)


def test_mul_py_scalar(left: "pd.Index[str]") -> None:
    """Test pd.Index[str] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 27), timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _06 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = f * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _16 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_py_sequence(left: "pd.Index[str]") -> None:
    """Test pd.Index[str] * Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 9, d) for d in (27, 28, 29)]
    d = [timedelta(seconds=s + 1) for s in range(3)]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _06 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = f * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _16 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]


def test_mul_numpy_array(left: "pd.Index[str]") -> None:
    """Test pd.Index[str] * numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array([np.datetime64(f"2025-09-{d}") for d in (27, 28, 29)], np.datetime64)
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * b, Never)
    check(assert_type(left * i, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * f, Never)
        assert_type(left * c, Never)
        assert_type(left * s, Never)
        assert_type(left * d, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Index` with the correct element type.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b * left, "npt.NDArray[np.bool_]")
    check(assert_type(i * left, "npt.NDArray[np.int64]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(f * left, "npt.NDArray[np.float64]")
        assert_type(c * left, "npt.NDArray[np.complex128]")
        assert_type(s * left, Any)
        assert_type(d * left, "npt.NDArray[np.timedelta64]")


def test_mul_pd_index(left: "pd.Index[str]") -> None:
    """Test pd.Index[str] * pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 9, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _06 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, "pd.Index[str]"), pd.Index, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _13 = f * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _16 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
