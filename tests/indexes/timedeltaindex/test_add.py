from datetime import (
    datetime,
    timedelta,
)
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check
from tests._typing import (
    np_ndarray_dt,
    np_ndarray_td,
)


@pytest.fixture
def left() -> pd.TimedeltaIndex:
    """Left operand"""
    lo = pd.Index([pd.Timedelta(1, "s")])  # left operand
    return check(assert_type(lo, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_add_py_scalar(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex + Python native scalars"""
    s = datetime(2025, 8, 20)
    d = timedelta(seconds=1)

    check(assert_type(left + s, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(left + d, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    check(assert_type(s + left, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(d + left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_add_py_sequence(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex + Python native sequences"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    check(assert_type(left + s, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(left + d, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    check(assert_type(s + left, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(d + left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_add_numpy_array(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex + numpy arrays"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    check(assert_type(left + s, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(left + d, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `DatetimeIndex` or `TimedeltaIndex`.
    check(assert_type(s + left, np_ndarray_dt), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(d + left, np_ndarray_td), pd.TimedeltaIndex, pd.Timedelta)


def test_add_pd_index(left: pd.TimedeltaIndex) -> None:
    """Test pd.TimedeltaIndex + pandas Indexes"""
    s = pd.Index([pd.Timestamp("2025-08-20")])
    d = pd.Index([pd.Timedelta(seconds=1)])

    check(assert_type(left + s, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(left + d, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)

    check(assert_type(s + left, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(d + left, pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)
