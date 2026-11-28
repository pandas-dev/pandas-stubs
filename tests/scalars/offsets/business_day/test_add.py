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
    CustomBusinessDay,
    Day,
    Hour,
    MonthEnd,
    Week,
)


@pytest.fixture
def left() -> BusinessDay:
    """Left operand"""
    return BusinessDay()


def test_add_py_scalar(left: BusinessDay) -> None:
    """Test BusinessDay + Python native scalars"""
    datum = date(2026, 10, 2)
    s = datetime(2026, 10, 2)
    d = timedelta(hours=2)

    check(assert_type(left + datum, pd.Timestamp), pd.Timestamp)
    check(assert_type(datum + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + d, BusinessDay), BusinessDay)
    check(assert_type(d + left, BusinessDay), BusinessDay)


def test_add_numpy_scalar(left: BusinessDay) -> None:
    """Test BusinessDay + numpy scalars"""
    s = np.datetime64("2026-10-02")
    d = np.timedelta64(2, "h")

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + d, BusinessDay), BusinessDay)
    check(assert_type(d + left, BusinessDay), BusinessDay)


def test_add_pd_scalar(left: BusinessDay) -> None:
    """Test BusinessDay + pandas scalars"""
    s = pd.Timestamp("2026-10-02")
    d = pd.Timedelta("2h")
    day = Day()
    h = Hour()
    w = Week()
    cbd = CustomBusinessDay()
    me = MonthEnd()

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)

    check(assert_type(left + d, BusinessDay), BusinessDay)
    check(assert_type(d + left, BusinessDay), BusinessDay)

    check(assert_type(left + day, BusinessDay), BusinessDay)
    check(assert_type(left + h, BusinessDay), BusinessDay)
    check(assert_type(left + w, BusinessDay), BusinessDay)

    result = check(assert_type(cbd + day, BusinessDay), BusinessDay)
    assert type(result) is BusinessDay

    if TYPE_CHECKING_INVALID_USAGE:
        _l_l = left + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]
        _l_me = left + me  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_numpy_array(left: BusinessDay) -> None:
    """Test BusinessDay + numpy arrays"""
    s = np.array([datetime(2026, 10, 2)], np.object_)

    if TYPE_CHECKING_INVALID_USAGE:

        def _l_s() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left + s, Never)

        # numpy's `__add__` wins `s + left` statically, so the reflected overload is
        # checked directly, in its own function since `Never` ends reachability.
        def _s_l() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left.__radd__(s), Never)
