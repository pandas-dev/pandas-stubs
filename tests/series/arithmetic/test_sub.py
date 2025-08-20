from datetime import (
    datetime,
    timedelta,
)
from typing import (
    TYPE_CHECKING,
    NoReturn,
)

import numpy as np
from numpy import typing as npt  # noqa: F401
import pandas as pd
from typing_extensions import assert_type

from tests import check

if TYPE_CHECKING:
    from pandas.core.series import TimedeltaSeries  # noqa: F401

left_i = pd.DataFrame({"a": [1, 2, 3]})["a"]  # left operand


def test_sub_py_scalar() -> None:
    """Test pd.Series[Any] - Python native scalars"""
    b, i, f, c = True, 1, 1.0, 1j

    check(assert_type(left_i - b, pd.Series), pd.Series)
    check(assert_type(left_i - i, pd.Series), pd.Series)
    check(assert_type(left_i - f, pd.Series), pd.Series)
    check(assert_type(left_i - c, pd.Series), pd.Series)

    check(assert_type(b - left_i, pd.Series), pd.Series)
    check(assert_type(i - left_i, pd.Series), pd.Series)
    check(assert_type(f - left_i, pd.Series), pd.Series)
    check(assert_type(c - left_i, pd.Series), pd.Series)

    check(assert_type(left_i.sub(b), pd.Series), pd.Series)
    check(assert_type(left_i.sub(i), pd.Series), pd.Series)
    check(assert_type(left_i.sub(f), pd.Series), pd.Series)
    check(assert_type(left_i.sub(c), pd.Series), pd.Series)

    check(assert_type(left_i.rsub(b), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(i), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(f), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(c), pd.Series), pd.Series)


def test_sub_py_sequence() -> None:
    """Test pd.Series[Any] - Python native sequence"""
    b, i, f, c = [True, False, True], [2, 3, 5], [1.0, 2.0, 3.0], [1j, 1j, 4j]

    check(assert_type(left_i - b, pd.Series), pd.Series)
    check(assert_type(left_i - i, pd.Series), pd.Series)
    check(assert_type(left_i - f, pd.Series), pd.Series)
    check(assert_type(left_i - c, pd.Series), pd.Series)

    check(assert_type(b - left_i, pd.Series), pd.Series)
    check(assert_type(i - left_i, pd.Series), pd.Series)
    check(assert_type(f - left_i, pd.Series), pd.Series)
    check(assert_type(c - left_i, pd.Series), pd.Series)

    check(assert_type(left_i.sub(b), pd.Series), pd.Series)
    check(assert_type(left_i.sub(i), pd.Series), pd.Series)
    check(assert_type(left_i.sub(f), pd.Series), pd.Series)
    check(assert_type(left_i.sub(c), pd.Series), pd.Series)

    check(assert_type(left_i.rsub(b), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(i), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(f), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(c), pd.Series), pd.Series)


def test_sub_numpy_array() -> None:
    """Test pd.Series[Any] - numpy array"""
    b = np.array([True, False, True], np.bool_)
    i = np.array([2, 3, 5], np.int64)
    f = np.array([1.0, 2.0, 3.0], np.float64)
    c = np.array([1.1j, 2.2j, 4.1j], np.complex128)

    check(assert_type(left_i - b, pd.Series), pd.Series)
    check(assert_type(left_i - i, pd.Series), pd.Series)
    check(assert_type(left_i - f, pd.Series), pd.Series)
    check(assert_type(left_i - c, pd.Series), pd.Series)

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rsub__` cannot override. At runtime, they return
    # `Series`s.
    # `mypy` thinks the return types are `Any`, which is a bug.
    check(assert_type(b - left_i, NoReturn), pd.Series)  # type: ignore[assert-type]
    check(
        assert_type(i - left_i, "npt.NDArray[np.int64]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(f - left_i, "npt.NDArray[np.float64]"), pd.Series  # type: ignore[assert-type]
    )
    check(
        assert_type(c - left_i, "npt.NDArray[np.complex128]"), pd.Series  # type: ignore[assert-type]
    )

    check(assert_type(left_i.sub(b), pd.Series), pd.Series)
    check(assert_type(left_i.sub(i), pd.Series), pd.Series)
    check(assert_type(left_i.sub(f), pd.Series), pd.Series)
    check(assert_type(left_i.sub(c), pd.Series), pd.Series)

    check(assert_type(left_i.rsub(b), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(i), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(f), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(c), pd.Series), pd.Series)


def test_sub_pd_series() -> None:
    """Test pd.Series[Any] - pandas series"""
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i - b, pd.Series), pd.Series)
    check(assert_type(left_i - i, pd.Series), pd.Series)
    check(assert_type(left_i - f, pd.Series), pd.Series)
    check(assert_type(left_i - c, pd.Series), pd.Series)

    check(assert_type(b - left_i, pd.Series), pd.Series)
    check(assert_type(i - left_i, pd.Series), pd.Series)
    check(assert_type(f - left_i, pd.Series), pd.Series)
    check(assert_type(c - left_i, pd.Series), pd.Series)

    check(assert_type(left_i.sub(b), pd.Series), pd.Series)
    check(assert_type(left_i.sub(i), pd.Series), pd.Series)
    check(assert_type(left_i.sub(f), pd.Series), pd.Series)
    check(assert_type(left_i.sub(c), pd.Series), pd.Series)

    check(assert_type(left_i.rsub(b), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(i), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(f), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(c), pd.Series), pd.Series)


anchor = datetime(2025, 8, 18)
left_ts = pd.DataFrame({"a": [anchor + timedelta(hours=h + 1) for h in range(3)]})["a"]


def test_sub_py_datetime() -> None:
    """Test pd.Series[Any] - Python native datetime"""
    s = anchor

    check(assert_type(left_ts - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)

    check(assert_type(s - left_ts, pd.Series), pd.Series, pd.Timedelta)

    check(assert_type(left_ts.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)

    check(assert_type(left_ts.rsub(s), pd.Series), pd.Series, pd.Timedelta)


def test_sub_numpy_datetime() -> None:
    """Test pd.Series[Any] - numpy datetime(s)"""
    s = np.datetime64(anchor)
    a = np.array([s + np.timedelta64(m, "m") for m in range(3)], np.datetime64)

    check(assert_type(left_ts - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left_ts - a, "TimedeltaSeries"), pd.Series, pd.Timedelta)

    check(assert_type(s - left_ts, pd.Series), pd.Series, pd.Timedelta)
    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rsub__` cannot override. At runtime, they return
    # `Series`s.
    check(assert_type(a - left_ts, "npt.NDArray[np.datetime64]"), pd.Series, pd.Timedelta)  # type: ignore[assert-type]

    check(assert_type(left_ts.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.sub(a), "TimedeltaSeries"), pd.Series, pd.Timedelta)

    check(assert_type(left_ts.rsub(s), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.rsub(a), pd.Series), pd.Series, pd.Timedelta)


def test_sub_pd_datetime() -> None:
    """Test pd.Series[Any] - Pandas datetime(s)"""
    s = pd.Timestamp(anchor)
    a = pd.Series([s + pd.Timedelta(minutes=m) for m in range(3)])

    check(assert_type(left_ts - s, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left_ts - a, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)

    check(assert_type(s - left_ts, pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(a - left_ts, pd.Series), pd.Series, pd.Timedelta)

    check(assert_type(left_ts.sub(s), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.sub(a), "TimedeltaSeries"), pd.Series, pd.Timedelta)

    check(assert_type(left_ts.rsub(s), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.rsub(a), pd.Series), pd.Series, pd.Timedelta)
