from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Never,
    assert_type,
)

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    np_ndarray_dt,
    np_ndarray_td,
)


@pytest.fixture
def left() -> pd.DatetimeIndex:
    """Left operand"""
    lo = pd.DatetimeIndex(["2025-08-20"])
    return check(assert_type(lo, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)


def test_sub_py_scalar(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - Python native scalars"""
    s = datetime(2025, 8, 20)
    d = timedelta(seconds=1)

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(s - left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_numpy_scalar(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - numpy scalars"""
    s = np.datetime64("2025-08-20")
    d = np.timedelta64(1, "s")

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(s - left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_pd_scalar(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - pandas scalars"""
    s = pd.Timestamp("2025-08-20")
    d = pd.Timedelta(seconds=1)

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(s - left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_py_sequence(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - Python native sequences"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(s - left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_numpy_scalar_sequence(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - sequences of numpy scalars"""
    s = [np.datetime64("2025-08-20")]
    d = [np.timedelta64(1, "s")]

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(s - left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        _1 = d - left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_pd_array(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - pandas extension arrays"""
    s = pd.array([datetime(2025, 8, 20)])
    d = pd.array([timedelta(seconds=1)])

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(s - left, pd.TimedeltaIndex, pd.Timedelta)


def test_sub_numpy_array(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - numpy arrays"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rsub__` cannot override. At runtime, `s - left`
    # returns a `TimedeltaIndex`, while `d - left` raises `TypeError`.
    check(assert_type(s - left, np_ndarray_dt), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(d - left, np_ndarray_td)


def test_sub_pd_index(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex - pandas Indexes"""
    s = pd.Index([pd.Timestamp("2025-08-20")])
    d = pd.Index([pd.Timedelta(seconds=1)])

    check(assert_type(left - s, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    check(assert_type(left - d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(s - left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(d - left, Never)
