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


def test_truediv_py_scalar(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 24), timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left / f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / d, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d / left, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.truediv(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left.truediv(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.truediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.div(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left.div(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.div(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rtruediv(b)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(i)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(f)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(c)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rtruediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rdiv(b)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(i)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(f)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(c)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rdiv(d), "pd.Series[float]"), pd.Series, np.floating)


def test_truediv_py_sequence(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] / Python native sequences"""
    b, i, f, c = [True], [2], [1.5], [1.7j]
    s, d = [datetime(2025, 9, 24)], [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left / f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / d, "pd.Series[float]"), pd.Series, float)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d / left, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.truediv(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left.truediv(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.truediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.div(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left.div(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.div(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rtruediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rtruediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rdiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rdiv(d), "pd.Series[float]"), pd.Series, np.floating)


def test_truediv_numpy_array(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] / numpy arrays"""
    b = np.array([True], np.bool_)
    i = np.array([2], np.int64)
    f = np.array([1.5], np.float64)
    c = np.array([1.7j], np.complex128)
    s = np.array([datetime(2025, 9, 24)], np.datetime64)
    d = np.array([timedelta(seconds=1)], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left / b, Never)
    check(assert_type(left / i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left / f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left / c, Never)
        assert_type(left / s, Never)
    check(assert_type(left / d, "pd.Series[float]"), pd.Series, np.floating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they lead to
    # errors or pd.Series.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b / left, "npt.NDArray[np.float64]")
        assert_type(i / left, "npt.NDArray[np.float64]")
        assert_type(f / left, "npt.NDArray[np.float64]")
        assert_type(c / left, "npt.NDArray[np.complex128]")
        assert_type(s / left, Any)
    check(assert_type(d / left, "npt.NDArray[np.float64]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.truediv(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left.truediv(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.truediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.div(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left.div(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.div(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rtruediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rtruediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rdiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rdiv(d), "pd.Series[float]"), pd.Series, np.floating)


def test_truediv_pd_scalar(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] / pandas scalars"""
    s, d = pd.Timestamp(2025, 9, 24), pd.Timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / d, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d / left, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.truediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rtruediv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rtruediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rdiv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rdiv(d), "pd.Series[float]"), pd.Series, np.floating)


def test_truediv_pd_index(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] / pandas Indexes"""
    b = pd.Index([True])
    i = pd.Index([2])
    f = pd.Index([1.5])
    c = pd.Index([1.7j])
    s, d = pd.Index([datetime(2025, 9, 24)]), pd.Index([timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left / f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / d, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d / left, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.truediv(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left.truediv(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.truediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.div(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left.div(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.div(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rtruediv(b)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(i)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(f)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(c)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rtruediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rdiv(b)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(i)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(f)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(c)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rdiv(d), "pd.Series[float]"), pd.Series, np.floating)


def test_truediv_pd_series(left: "pd.Series[pd.Timedelta]") -> None:
    """Test pd.Series[pd.Timedelta] / pandas Series"""
    b = pd.Series([True])
    i = pd.Series([2])
    f = pd.Series([1.5])
    c = pd.Series([1.7j])
    s, d = pd.Series([datetime(2025, 9, 24)]), pd.Series([timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / i, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left / f, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(left / d, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    check(assert_type(d / left, "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(
        assert_type(left.truediv(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left.truediv(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    if TYPE_CHECKING_INVALID_USAGE:
        left.truediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.truediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.truediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.div(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(i), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left.div(f), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.div(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.div(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.div(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rtruediv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rtruediv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rtruediv(d), "pd.Series[float]"), pd.Series, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        left.rdiv(b)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(i)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(f)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(c)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rdiv(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.rdiv(d), "pd.Series[float]"), pd.Series, np.floating)
