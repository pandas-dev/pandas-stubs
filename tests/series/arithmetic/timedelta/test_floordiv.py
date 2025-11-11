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
def left() -> "pd.Series[pd.Timedelta]":
    """Left operand"""
    lo = pd.Series([pd.Timedelta(1, "s")])
    return check(assert_type(lo, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_floordiv_py_scalar(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] // Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 24), timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left // f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // d, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.floordiv(i), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(left.floordiv(f), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(d), "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(b)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(i)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(f)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(c)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(d), "pd.Series[int]"), pd.Series, np.integer)


def test_floordiv_py_sequence(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] // Python native sequences"""
    b, i, f, c = [True], [2], [1.5], [1.7j]
    s, d = [datetime(2025, 9, 24)], [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left // f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // d, "pd.Series[int]"), pd.Series, int)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.floordiv(i), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(left.floordiv(f), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(d), "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(d), "pd.Series[int]"), pd.Series, np.integer)


def test_floordiv_numpy_array(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] // numpy arrays"""
    b = np.array([True], np.bool_)
    i = np.array([2], np.int64)
    f = np.array([1.5], np.float64)
    c = np.array([1.7j], np.complex128)
    s = np.array([datetime(2025, 9, 24)], np.datetime64)
    d = np.array([timedelta(seconds=1)], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left // b, Never)
    check(assert_type(left // i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left // f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left // c, Never)
        assert_type(left // s, Never)
    check(assert_type(left // d, "pd.Series[int]"), pd.Series, np.integer)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rfloordiv__` cannot override. At runtime, they lead to
    # errors or pd.Series.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b // left, "npt.NDArray[np.int8]")
        assert_type(i // left, "npt.NDArray[np.int64]")
        assert_type(f // left, "npt.NDArray[np.float64]")
        assert_type(c // left, Any)
        assert_type(s // left, Any)
    check(assert_type(d // left, "npt.NDArray[np.int64]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.floordiv(i), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(left.floordiv(f), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(d), "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(d), "pd.Series[int]"), pd.Series, np.integer)


def test_floordiv_pd_scalar(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] // pandas scalars"""
    s, d = pd.Timestamp(2025, 9, 24), pd.Timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // d, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(d), "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(d), "pd.Series[int]"), pd.Series, np.integer)


def test_floordiv_pd_index(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] // pandas Indexes"""
    b = pd.Index([True])
    i = pd.Index([2])
    f = pd.Index([1.5])
    c = pd.Index([1.7j])
    s, d = pd.Index([datetime(2025, 9, 24)]), pd.Index([timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left // f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // d, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.floordiv(i), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(left.floordiv(f), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(d), "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(d), "pd.Series[int]"), pd.Series, np.integer)


def test_floordiv_pd_series(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] // pandas Series"""
    b = pd.Series([True])
    i = pd.Series([2])
    f = pd.Series([1.5])
    c = pd.Series([1.7j])
    s, d = pd.Series([datetime(2025, 9, 24)]), pd.Series([timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left // b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left // f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left // c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left // s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left // d, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s // left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d // left, "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.floordiv(i), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    check(
        assert_type(left.floordiv(f), "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.floordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.floordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.floordiv(d), "pd.Series[int]"), pd.Series, np.integer)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rfloordiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rfloordiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rfloordiv(d), "pd.Series[int]"), pd.Series, np.integer)
