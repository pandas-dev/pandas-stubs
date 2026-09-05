from datetime import (
    datetime,
    timedelta,
)
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import Day


@pytest.fixture
def left() -> pd.PeriodIndex:
    """Left operand"""
    lo = pd.PeriodIndex(["2025-08-20"], freq="D")
    return check(assert_type(lo, pd.PeriodIndex), pd.PeriodIndex, pd.Period)


def test_sub_py_scalar(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex - Python native scalars"""
    d = timedelta(days=1)
    i = 1
    s = datetime(2025, 8, 20)

    check(assert_type(left - d, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(left - i, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left - s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_numpy_scalar(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex - numpy scalars"""
    d = np.timedelta64(1, "D")
    i = np.int64(1)
    s = np.datetime64("2025-08-20")

    check(assert_type(left - d, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(left - i, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left - s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation]


def test_sub_pd_scalar(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex - pandas scalars"""
    d = pd.Timedelta(days=1)
    off = Day(1)
    p = pd.Period("2025-08-20", freq="D")

    check(assert_type(left - d, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(left - off, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(left - p, pd.Index), pd.Index)
    check(assert_type(p - left, pd.Index), pd.Index)


def test_sub_pd_index(left: pd.PeriodIndex) -> None:
    """Test pd.PeriodIndex - pandas Indexes"""
    pi = pd.PeriodIndex(["2025-08-20"], freq="D")
    ti = pd.TimedeltaIndex([pd.Timedelta(days=1)])

    check(assert_type(left - pi, pd.Index), pd.Index)
    check(assert_type(left - ti, pd.PeriodIndex), pd.PeriodIndex, pd.Period)
