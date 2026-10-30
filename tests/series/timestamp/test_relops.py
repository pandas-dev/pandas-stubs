from __future__ import annotations

from datetime import datetime
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[pd.Timestamp]:
    """Left operand"""
    lo = pd.Series([pd.Timestamp("2022-07-31 08:35:12")])
    return check(assert_type(lo, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_relops_py_scalar(left: pd.Series[pd.Timestamp]) -> None:
    """Test pd.Series[pd.Timestamp] <comparison> py scalar"""
    # GH 175
    dtdt = datetime(2022, 7, 31, 8, 32, 21)
    ts = pd.Timestamp("2022-07-31 08:35:12")

    check(assert_type(left < dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= dtdt, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= ts, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(dtdt), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(dtdt), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(dtdt), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(dtdt), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(ts), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(ts), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(ts), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(ts), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence(left: pd.Series[pd.Timestamp]) -> None:
    values = [pd.Timestamp("2022-08-01")]

    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_datetime_sequence(left: pd.Series[pd.Timestamp]) -> None:
    values = [datetime(2022, 8, 1)]

    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array(left: pd.Series[pd.Timestamp]) -> None:
    array = np.array([np.datetime64("2022-08-01", "ns")], dtype=np.datetime64)

    check(assert_type(left < array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > array, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= array, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.__lt__(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__le__(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__gt__(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__ge__(array), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(array), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(array), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_index(left: pd.Series[pd.Timestamp]) -> None:
    index = pd.Index(left)
    datetime_index = pd.DatetimeIndex(left)
    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= index, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < datetime_index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= datetime_index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > datetime_index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= datetime_index, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(datetime_index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(datetime_index <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(datetime_index > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(datetime_index >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(index), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(datetime_index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(datetime_index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(datetime_index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(datetime_index), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_series(left: pd.Series[pd.Timestamp]) -> None:
    other = left.copy()
    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)
