from datetime import (
    datetime,
    timedelta,
)
from typing import Any

import numpy as np
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
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
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
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
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
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_py_sequence() -> None:
    """Test pd.Series[pd.Timestamp] + Python native sequences"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        # Series[Timestamp] + Sequence[timedelta] should work, see pandas-dev/pandas#62353
        _a = left + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _b = d + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_numpy_array() -> None:
    """Test pd.Series[pd.Timestamp] + numpy arrays"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left + s, Never)
    check(assert_type(left + d, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(s + left, Any)  # pyright: ignore[reportAssertTypeFailure]
    # Here even the dtype of `NDArray` is in the wrong direction.
    # `np.datetime64` would be more sensible.
    check(
        assert_type(d + left, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
        pd.Timestamp,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        left.add(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.add(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        left.radd(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
    check(assert_type(left.radd(d), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_add_pd_index() -> None:
    """Test pd.Series[pd.Timestamp] + pandas Indexes"""
    s = pd.Index([pd.Timestamp("2025-08-20")])
    d = pd.Index([pd.Timedelta(seconds=1)])

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
