from datetime import (
    datetime,
    timedelta,
)
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

if TYPE_CHECKING:
    from pandas.core.series import TimedeltaSeries  # noqa: F401

left = pd.Series([pd.Timestamp(2025, 8, 20)])  # left operand


def test_sub_py_scalar() -> None:
    """Test pd.Series[pd.Timestamp] - Python native scalars"""
    s = datetime(2025, 8, 20)
    d = timedelta(seconds=1)

    check(assert_type(left - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(s - left, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _ = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(left.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(left.rsub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_sub_numpy_scalar() -> None:
    """Test pd.Series[pd.Timestamp] - numpy scalars"""
    s = np.datetime64("2025-08-20")
    d = np.timedelta64(1, "s")

    check(assert_type(left - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(s - left, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _ = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(left.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(left.rsub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_sub_pd_scalar() -> None:
    """Test pd.Series[pd.Timestamp] - pandas scalars"""
    s = pd.Timestamp("2025-08-20")
    d = pd.Timedelta(seconds=1)

    check(assert_type(left - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(s - left, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _ = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(left.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(left.rsub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_sub_py_sequence() -> None:
    """Test pd.Series[pd.Timestamp] - Python native sequence"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left - s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _a = left - d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _1 = s - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _b = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        left.sub(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.sub(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]

        left.rsub(s)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]
        left.rsub(d)  # type: ignore[call-overload] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_sub_numpy_array() -> None:
    """Test pd.Series[pd.Timestamp] - numpy array"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    check(assert_type(left - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`s.
    check(assert_type(s - left, "npt.NDArray[np.datetime64]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(d - left, "npt.NDArray[np.timedelta64]")

    check(assert_type(left.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(left.rsub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]


def test_sub_pd_series() -> None:
    """Test pd.Series[pd.Timestamp] - pandas Series"""
    s = pd.Series([pd.Timestamp("2025-08-20")])
    d = pd.Series([pd.Timedelta(seconds=1)])

    check(assert_type(left - s, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(s - left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _ = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(left.sub(s), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left - d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    check(assert_type(left.rsub(s), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        left.rsub(d)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
