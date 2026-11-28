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
    Micro,
    Milli,
    Minute,
    MonthEnd,
    Nano,
    Second,
    Tick,
    Week,
)


@pytest.fixture
def left() -> Tick:
    """Left operand"""
    return Hour()


def test_add_py_scalar(left: Tick) -> None:
    """Test Tick + Python native scalars"""
    datum = date(2026, 10, 2)
    s = datetime(2026, 10, 2)
    d = timedelta(hours=2)

    check(assert_type(left + datum, pd.Timestamp), pd.Timestamp)
    check(assert_type(datum + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + d, pd.Timedelta), pd.Timedelta)
    check(assert_type(d + left, pd.Timedelta), pd.Timedelta)


def test_add_numpy_scalar(left: Tick) -> None:
    """Test Tick + numpy scalars"""
    s = np.datetime64("2026-10-02")
    d = np.timedelta64(2, "h")

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + d, pd.Timedelta), pd.Timedelta)
    check(assert_type(d + left, pd.Timedelta), pd.Timedelta)


def test_add_pd_scalar(left: Tick) -> None:
    """Test Tick + pandas scalars"""
    s = pd.Timestamp("2026-10-02")
    d = pd.Timedelta("2h")
    h = Hour(2)
    m = Minute(2)
    sec = Second(2)
    ms = Milli(2)
    us = Micro(2)
    ns = Nano(2)
    m30 = Minute(30)
    m60 = Minute(60)
    day = Day()
    w = Week()
    bd = BusinessDay()
    me = MonthEnd()

    check(assert_type(left + s, pd.Timestamp), pd.Timestamp)
    check(assert_type(s + left, pd.Timestamp), pd.Timestamp)

    check(assert_type(left + pd.NaT, NaTType), NaTType)
    check(assert_type(pd.NaT + left, NaTType), NaTType)

    check(assert_type(left + d, pd.Timedelta), pd.Timedelta)
    check(assert_type(d + left, pd.Timedelta), pd.Timedelta)

    check(assert_type(h + h, Hour), Hour)
    check(assert_type(m + m, Minute), Minute)
    check(assert_type(sec + sec, Second), Second)
    check(assert_type(ms + ms, Milli), Milli)
    check(assert_type(us + us, Micro), Micro)
    check(assert_type(ns + ns, Nano), Nano)

    # At runtime these give `Minute(150)` and `Hour(3)`.
    check(assert_type(h + m30, Tick), Tick)
    check(assert_type(h + m60, Tick), Tick)

    check(assert_type(left + day, pd.Timedelta), pd.Timedelta)
    check(assert_type(left + w, pd.Timedelta), pd.Timedelta)
    check(assert_type(left + bd, BusinessDay), BusinessDay)

    if TYPE_CHECKING_INVALID_USAGE:
        _l_me = left + me  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType] # pyrefly: ignore[unsupported-operation] # ty: ignore[unsupported-operator]


def test_add_numpy_array(left: Tick) -> None:
    """Test Tick + numpy arrays"""
    s = np.array([datetime(2026, 10, 2)], np.object_)

    if TYPE_CHECKING_INVALID_USAGE:

        def _l_s() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left + s, Never)

        # numpy's `__add__` wins `s + left` statically, so the reflected overload is
        # checked directly, in its own function since `Never` ends reachability.
        def _s_l() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(left.__radd__(s), Never)
