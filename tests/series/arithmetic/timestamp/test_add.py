from datetime import (
    datetime,
    timedelta,
)

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

left = pd.Series([pd.Timestamp(2025, 8, 20)])  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Series[pd.Timestamp] + Python native scalars"""
    s = datetime(2025, 8, 20)
    d = timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_numpy_scalar() -> None:
    """Test pd.Series[pd.Timestamp] + numpy scalars"""
    s = np.datetime64("2025-08-20")
    d = np.timedelta64(1, "s")

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_pd_scalar() -> None:
    """Test pd.Series[pd.Timestamp] + pandas scalars"""
    s = pd.Timestamp("2025-08-20")
    d = pd.Timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_py_sequence() -> None:
    """Test pd.Series[pd.Timestamp] + Python native sequence"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _a = left + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _b = d + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        left.add(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.add(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.radd(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.radd(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_add_numpy_array() -> None:
    """Test pd.Series[pd.Timestamp] + numpy array"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left + s, Never)
    check(assert_type(left + d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(s + left, "npt.NDArray[np.datetime64]")
    # Here even the dtype of `NDArray` is in the wrong direction.
    # `np.datetime64` would be more sensible.
    check(assert_type(d + left, "npt.NDArray[np.timedelta64]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_pd_series() -> None:
    """Test pd.Series[pd.Timestamp] + pandas Series"""
    s = pd.Series([pd.Timestamp("2025-08-20")])
    d = pd.Series([pd.Timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(left + d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(d + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
