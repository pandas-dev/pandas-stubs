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
def left() -> "pd.Series[str]":
    """left operand"""
    lo = pd.Series(["1", "2", "3"])
    return check(assert_type(lo, "pd.Series[str]"), pd.Series, str)


def test_mul_py_scalar(left: "pd.Series[str]") -> None:
    """Test pd.Series[str] * Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 27), timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _02 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _12 = f * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_mul_py_sequence(left: "pd.Series[str]") -> None:
    """Test pd.Series[str] * Python native sequences"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]
    s = [datetime(2025, 9, d) for d in (27, 28, 29)]
    d = [timedelta(seconds=s + 1) for s in range(3)]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _02 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _12 = f * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_mul_numpy_array(left: "pd.Series[str]") -> None:
    """Test pd.Series[str] * numpy arrays"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)
    s = np.array([np.datetime64(f"2025-09-{d}") for d in (27, 28, 29)], np.datetime64)
    d = np.array([np.timedelta64(s + 1, "s") for s in range(3)], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * b, Never)
    check(assert_type(left * i, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left * f, Never)
        assert_type(left * c, Never)
        assert_type(left * s, Never)
        assert_type(left * d, Never)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rmul__` cannot override. At runtime, they return
    # `Series` with the correct element type.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b * left, "npt.NDArray[np.bool_]")
    check(assert_type(i * left, "npt.NDArray[np.int64]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(f * left, "npt.NDArray[np.float64]")
        assert_type(c * left, "npt.NDArray[np.complex128]")
        assert_type(s * left, Any)
        assert_type(d * left, "npt.NDArray[np.timedelta64]")

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_mul_pd_index(left: "pd.Series[str]") -> None:
    """Test pd.Series[str] * pandas Indexes"""
    b = pd.Index([True, False, True])
    i = pd.Index([2, 3, 5])
    f = pd.Index([1.0, 2.0, 3.0])
    c = pd.Index([1.1j, 2.2j, 4.1j])
    s = pd.Index([datetime(2025, 9, d) for d in (27, 28, 29)])
    d = pd.Index([timedelta(seconds=s + 1) for s in range(3)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _02 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator,type-var,unused-ignore] # pyright: ignore[reportOperatorIssue] # mypy gives different errors for mypy and test_dist
    check(assert_type(i * left, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _12 = f * left  # type: ignore[operator,type-var,unused-ignore] # pyright: ignore[reportOperatorIssue] # mypy gives different errors for mypy and test_dist
        _13 = c * left  # type: ignore[operator,type-var,unused-ignore] # pyright: ignore[reportOperatorIssue] # mypy gives different errors for mypy and test_dist
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_mul_pd_series(left: "pd.Series[str]") -> None:
    """Test pd.Series[str] * pandas Series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])
    s = pd.Series([datetime(2025, 9, d) for d in (27, 28, 29)])
    d = pd.Series([timedelta(seconds=s + 1) for s in range(3)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left * b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left * i, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _02 = left * f  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _03 = left * c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left * s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _05 = left * d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(i * left, "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        _12 = f * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _15 = d * left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.mul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.mul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.mul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rmul(i), "pd.Series[str]"), pd.Series, str)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rmul(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rmul(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
