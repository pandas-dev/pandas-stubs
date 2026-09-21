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
    lo = pd.DatetimeIndex(["2025-08-20"])  # left operand
    return check(assert_type(lo, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)


def test_add_py_scalar(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex + Python native scalars"""
    s = datetime(2025, 8, 20)
    d = timedelta(seconds=1)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(left + d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(d + left, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)


def test_add_py_sequence(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex + Python native sequences"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(left + d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(d + left, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)


def test_add_numpy_array(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex + numpy arrays"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(left + s, Never)
    check(assert_type(left + d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `DatetimeIndex`.
    if TYPE_CHECKING_INVALID_USAGE:
        assert_type(s + left, np_ndarray_dt)
    check(assert_type(d + left, np_ndarray_td), pd.DatetimeIndex, pd.Timestamp)


def test_add_pd_index(left: pd.DatetimeIndex) -> None:
    """Test pd.DatetimeIndex + pandas Indexes"""
    s = pd.Index([pd.Timestamp("2025-08-20")])
    d = pd.Index([pd.Timedelta(seconds=1)])

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(left + d, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    if TYPE_CHECKING_INVALID_USAGE:
        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
    check(assert_type(d + left, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
