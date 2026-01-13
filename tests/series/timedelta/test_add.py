from datetime import (
    datetime,
    timedelta,
)

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    np_ndarray_dt,
    np_ndarray_td,
)

left = pd.Series([pd.Timedelta(1, "s")])  # left operand


def test_add_py_scalar() -> None:
    """Test pd.Series[pd.Timedelta] + Python native scalars"""
    s = datetime(2025, 8, 20)
    d = timedelta(seconds=1)

    check(assert_type(left + s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left + d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(s + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(d + left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_add_numpy_scalar() -> None:
    """Test pd.Series[pd.Timedelta] + numpy scalars"""
    s = np.datetime64("2025-08-20")
    d = np.timedelta64(1, "s")

    check(assert_type(left + s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left + d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(s + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(d + left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_add_pd_scalar() -> None:
    """Test pd.Series[pd.Timedelta] + pandas scalars"""
    s = pd.Timestamp("2025-08-20")
    d = pd.Timedelta(seconds=1)

    check(assert_type(left + s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left + d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(s + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(d + left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_add_py_sequence() -> None:
    """Test pd.Series[pd.Timedelta] + Python native sequences"""
    s = [datetime(2025, 8, 20)]
    d = [timedelta(seconds=1)]

    if TYPE_CHECKING_INVALID_USAGE:
        # Series[Timedelta] + Sequence[datetime] should work, see pandas-dev/pandas#62353
        _0 = left + s  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        # Series[Timedelta] + Sequence[timedelta] should work, see pandas-dev/pandas#62353
        _a = left + d  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

        _1 = s + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _b = d + left  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_add_numpy_array() -> None:
    """Test pd.Series[pd.Timedelta] + numpy arrays"""
    s = np.array([np.datetime64("2025-08-20")], np.datetime64)
    d = np.array([np.timedelta64(1, "s")], np.timedelta64)

    check(assert_type(left + s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left + d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__radd__` cannot override. At runtime, they return
    # `Series`.
    check(assert_type(s + left, np_ndarray_dt), pd.Series, pd.Timestamp)
    check(assert_type(d + left, np_ndarray_td), pd.Series, pd.Timedelta)

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_add_pd_index() -> None:
    """Test pd.Series[pd.Timedelta] + pandas Indexes"""
    s = pd.Index([pd.Timestamp("2025-08-20")])
    d = pd.Index([pd.Timedelta(seconds=1)])

    check(assert_type(left + s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left + d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(s + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(d + left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_add_pd_series() -> None:
    """Test pd.Series[pd.Timedelta] + pandas Series"""
    s = pd.Series([pd.Timestamp("2025-08-20")])
    d = pd.Series([pd.Timedelta(seconds=1)])

    check(assert_type(left + s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left + d, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(s + left, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(d + left, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.add(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.add(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(left.radd(s), "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
    check(assert_type(left.radd(d), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
