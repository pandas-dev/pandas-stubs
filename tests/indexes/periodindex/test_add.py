from datetime import timedelta
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    np_ndarray_int64,
    np_ndarray_td,
)


@pytest.fixture
def left() -> pd.PeriodIndex:
    """Left operand"""
    lo = pd.PeriodIndex(["2025-08-20"], freq="D")  # left operand
    return check(assert_type(lo, pd.PeriodIndex), pd.PeriodIndex, pd.Period)


def test_add_py_scalar(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex + Python native scalars"""
    d = timedelta(days=1)
    i = 1
    p = pd.Period("2025-08-20", freq="D")

    check(assert_type(left + d, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(d + left, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    check(assert_type(left + i, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(i + left, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_add_py_sequence(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex + Python native sequences"""
    d = [timedelta(days=1)]
    i = [1]
    p = [pd.Period("2025-08-20", freq="D")]

    check(assert_type(left + d, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(d + left, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    check(assert_type(left + i, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(i + left, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_add_numpy_array(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex + numpy arrays"""
    d = np.array([np.timedelta64(1, "D")], np.timedelta64)
    i = np.array([1], np.int64)

    check(assert_type(left + d, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(left + i, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `PeriodIndex`.
    check(assert_type(d + left, np_ndarray_td), pd.PeriodIndex, pd.Period)
    check(assert_type(i + left, np_ndarray_int64), pd.PeriodIndex, pd.Period)


def test_add_pd_index(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex + pandas Indexes"""
    td = pd.TimedeltaIndex(["1D"])
    ii = pd.Index([1])
    p = pd.PeriodIndex(["2025-08-20"], freq="D")

    check(assert_type(left + td, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(td + left, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    check(assert_type(left + ii, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(ii + left, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left + p  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
        _1 = p + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]
