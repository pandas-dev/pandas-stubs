from __future__ import annotations

from datetime import timedelta
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check
from tests._typing import np_ndarray_td


@pytest.fixture
def left() -> pd.Series[pd.Timedelta]:
    """Left operand"""
    lo = pd.Series([pd.Timedelta(seconds=10)])
    return check(assert_type(lo, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_relops_py_scalar(left: pd.Series[pd.Timedelta]) -> None:
    """Test pd.Series[pd.Timedelta] <comparison> py scalar"""
    # GH 175
    dttd = timedelta(seconds=10)
    td = pd.Timedelta(seconds=10)

    check(assert_type(left < dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= dttd, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left < td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= td, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(dttd), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(dttd), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(dttd), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(dttd), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(td), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(td), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(td), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(td), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_sequence(left: pd.Series[pd.Timedelta]) -> None:
    values = [pd.Timedelta(days=1)]

    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_timedelta_sequence(left: pd.Series[pd.Timedelta]) -> None:
    values = [timedelta(days=1)]

    check(assert_type(left < values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > values, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= values, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(values), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(values), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array(left: pd.Series[pd.Timedelta]) -> None:
    array: np_ndarray_td = np.array([1], dtype="timedelta64[D]")

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


def test_relops_pd_index(left: pd.Series[pd.Timedelta]) -> None:
    index = pd.Index(left)

    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= index, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(index), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_series(left: pd.Series[pd.Timedelta]) -> None:
    other = left.copy()

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)
