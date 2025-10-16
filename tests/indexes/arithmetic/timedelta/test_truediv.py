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
def left() -> "pd.Index[pd.Timedelta]":
    """Left operand"""
    # pandas-dev/pandas#62524: An index of Python native timedeltas can be
    # produced, instead of a TimedeltaIndex, hence this test file
    lo = pd.Index([1]) * [timedelta(seconds=1)]  # left operand
    return check(assert_type(lo, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)


def test_truediv_py_scalar(left: "pd.Index[pd.Timedelta]") -> None:
    """Test "pd.Index[pd.Timedelta]" / Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j
    s, d = datetime(2025, 9, 24), timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left / f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / d, "pd.Index[float]"), pd.Index, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    # TODO: pandas-dev/pandas#62712
    # check(assert_type(d / left, "pd.Index[float]"), pd.Index, np.floating)


def test_truediv_py_sequence(left: "pd.Index[pd.Timedelta]") -> None:
    """Test "pd.Index[pd.Timedelta]" / Python native sequences"""
    b, i, f, c = [True], [2], [1.5], [1.7j]
    s, d = [datetime(2025, 9, 24)], [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left / f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / d, "pd.Index[float]"), pd.Index, float)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d / left, "pd.Index[float]"), pd.Index, float)


def test_truediv_numpy_array(left: "pd.Index[pd.Timedelta]") -> None:
    """Test "pd.Index[pd.Timedelta]" / numpy arrays"""
    b = np.array([True], np.bool_)
    i = np.array([2], np.int64)
    f = np.array([1.5], np.float64)
    c = np.array([1.7j], np.complex128)
    s = np.array([datetime(2025, 9, 24)], np.datetime64)
    d = np.array([timedelta(seconds=1)], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left / b, Never)
    check(assert_type(left / i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left / f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left / c, Never)
        assert_type(left / s, Never)
    check(assert_type(left / d, "pd.Index[float]"), pd.Index, np.floating)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rtruediv__` cannot override. At runtime, they lead to
    # errors or pd.Series.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(b / left, "npt.NDArray[np.float64]")
        assert_type(i / left, "npt.NDArray[np.float64]")
        assert_type(f / left, "npt.NDArray[np.float64]")
        assert_type(c / left, "npt.NDArray[np.complex128]")
        assert_type(s / left, Any)
    check(assert_type(d / left, "npt.NDArray[np.float64]"), pd.Index, float)


def test_truediv_pd_scalar(left: "pd.Index[pd.Timedelta]") -> None:
    """Test "pd.Index[pd.Timedelta]" / pandas scalars"""
    s, d = pd.Timestamp(2025, 9, 24), pd.Timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / d, "pd.Index[float]"), pd.Index, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    # TODO: pandas-dev/pandas#62712
    # check(assert_type(d / left, "pd.Index[float]"), pd.Index, np.floating)


def test_truediv_pd_index(left: "pd.Index[pd.Timedelta]") -> None:
    """Test "pd.Index[pd.Timedelta]" / pandas pd.Indexes"""
    b = pd.Index([True])
    i = pd.Index([2])
    f = pd.Index([1.5])
    c = pd.Index([1.7j])
    s, d = pd.Index([datetime(2025, 9, 24)]), pd.Index([timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _00 = left / b  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / i, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    check(assert_type(left / f, "pd.Index[pd.Timedelta]"), pd.Index, timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _03 = left / c  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _04 = left / s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left / d, "pd.Index[float]"), pd.Index, np.floating)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = b / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _12 = f / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _13 = c / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _14 = s / left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d / left, "pd.Index[float]"), pd.Index, float)
