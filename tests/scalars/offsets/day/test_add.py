from datetime import (
    date,
    datetime,
    timedelta,
)
from typing import (
    Never,
    assert_type,
)

import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytest

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.offsets import (
    BusinessDay,
    Day,
    Hour,
    MonthEnd,
    Week,
)


@pytest.fixture
def left() -> Day:
    """Left operand"""
    return Day()


def test_add_py_scalar(left: Day) -> None:
    """Test Day + Python native scalars"""
    datum = date(2026, 10, 2)
    s = datetime(2026, 10, 2)
    d = timedelta(hours=2)

    check(assert_type(left + datum, pd.Timestamp), pd.Timestamp)
    check(assert_type(datum + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + d, timedelta), timedelta)
    check(assert_type(d + left, timedelta), timedelta)


def test_add_numpy_scalar(left: Day) -> None:
    """Test Day + numpy scalars"""
    s = np.datetime64("2026-10-02")
    d = np.timedelta64(2, "h")

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + d, np.timedelta64), np.timedelta64)
    check(assert_type(d + left, np.timedelta64), np.timedelta64)


def test_add_pd_scalar(left: Day) -> None:
    """Test Day + pandas scalars"""
    s = pd.Timestamp("2026-10-02")
    d = pd.Timedelta("2h")
    day = Day(2)
    h = Hour()
    w = Week()
    bd = BusinessDay()
    me = MonthEnd()

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)

    check(assert_type(left + d, pd.Timedelta), pd.Timedelta)
    check(assert_type(d + left, pd.Timedelta), pd.Timedelta)

    check(assert_type(left + day, Day), Day)
    check(assert_type(left + h, pd.Timedelta), pd.Timedelta)
    check(assert_type(left + w, timedelta), timedelta)
    check(assert_type(left + bd, BusinessDay), BusinessDay)

    if TYPE_CHECKING_INVALID_USAGE:
        _l_me = left + me  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_numpy_array(left: Day) -> None:
    """Test Day + numpy arrays"""
    s = np.array([datetime(2026, 10, 2)], np.object_)

    if TYPE_CHECKING_INVALID_USAGE:

        def _l_s() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left + s, Never)

        # numpy's `__add__` wins `s + left` statically, so the reflected overload is
        # checked directly, in its own function since `Never` ends reachability.
        def _s_l() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left.__radd__(s), Never)
