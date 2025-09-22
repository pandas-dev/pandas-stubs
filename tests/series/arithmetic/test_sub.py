from datetime import (
    datetime,
    timedelta,
)
from typing import Any

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

anchor = datetime(2025, 8, 18)

# left operands
left_i = pd.DataFrame({"a": [1, 2, 3]})["a"]
left_ts = pd.DataFrame({"a": [anchor + timedelta(hours=h + 1) for h in range(3)]})["a"]
left_td = pd.DataFrame({"a": [timedelta(hours=h, minutes=1) for h in range(3)]})["a"]


def test_sub_i_py_scalar() -> None:
    """Test pd.Series[Any] (int) - Python native scalars"""
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


def test_sub_i_py_sequence() -> None:
    """Test pd.Series[Any] (int) - Python native sequences"""
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


def test_sub_i_numpy_array() -> None:
    """Test pd.Series[Any] (int) - numpy arrays"""
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
    # `Series`.
    # microsoft/pyright#10924
    check(
        assert_type(b - left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(i - left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(f - left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )
    check(
        assert_type(c - left_i, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
    )

    check(assert_type(left_i.sub(b), pd.Series), pd.Series)
    check(assert_type(left_i.sub(i), pd.Series), pd.Series)
    check(assert_type(left_i.sub(f), pd.Series), pd.Series)
    check(assert_type(left_i.sub(c), pd.Series), pd.Series)

    check(assert_type(left_i.rsub(b), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(i), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(f), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(c), pd.Series), pd.Series)


def test_sub_i_pd_series() -> None:
    """Test pd.Series[Any] (int) - pandas series"""
    a = pd.DataFrame({"a": [1, 2, 3]})["a"]
    b = pd.Series([True, False, True])
    i = pd.Series([2, 3, 5])
    f = pd.Series([1.0, 2.0, 3.0])
    c = pd.Series([1.1j, 2.2j, 4.1j])

    check(assert_type(left_i - a, pd.Series), pd.Series)
    check(assert_type(left_i - b, pd.Series), pd.Series)
    check(assert_type(left_i - i, pd.Series), pd.Series)
    check(assert_type(left_i - f, pd.Series), pd.Series)
    check(assert_type(left_i - c, pd.Series), pd.Series)

    check(assert_type(a - left_i, pd.Series), pd.Series)
    check(assert_type(b - left_i, pd.Series), pd.Series)
    check(assert_type(i - left_i, pd.Series), pd.Series)
    check(assert_type(f - left_i, pd.Series), pd.Series)
    check(assert_type(c - left_i, pd.Series), pd.Series)

    check(assert_type(left_i.sub(a), pd.Series), pd.Series)
    check(assert_type(left_i.sub(b), pd.Series), pd.Series)
    check(assert_type(left_i.sub(i), pd.Series), pd.Series)
    check(assert_type(left_i.sub(f), pd.Series), pd.Series)
    check(assert_type(left_i.sub(c), pd.Series), pd.Series)

    check(assert_type(left_i.rsub(a), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(b), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(i), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(f), pd.Series), pd.Series)
    check(assert_type(left_i.rsub(c), pd.Series), pd.Series)


def test_sub_ts_py_datetime() -> None:
    """Test pd.Series[Any] (Timestamp | Timedelta) - Python native datetime"""
    s = anchor
    a = [s + timedelta(minutes=m) for m in range(3)]

    check(assert_type(left_ts - s, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        # Series[Any] (Timestamp) - Sequence[datetime] should work, see pandas-dev/pandas#62353
        _1 = left_ts - a  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    # Series[Any] (Timedelta) - datetime fails at runtime,
    # which cannot be revealed by our static type checking
    # _2 = left_td - s
    if TYPE_CHECKING_INVALID_USAGE:
        # Series[Any] (Timedelta) - Sequence[datetime] is not supported by Pandas,
        # see pandas-dev/pandas#62353. Even if such __sub__ is supported
        # it will fail at runtime here,
        # which cannot be revealed by our static type checking
        _3 = left_td - a  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(assert_type(s - left_ts, pd.Series), pd.Series, pd.Timedelta)
    if TYPE_CHECKING_INVALID_USAGE:
        # Sequence[datetime] - Series[Any] (Timestamp) should work, see pandas-dev/pandas#62353
        _5 = a - left_ts  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
    check(assert_type(s - left_td, pd.Series), pd.Series, pd.Timestamp)
    if TYPE_CHECKING_INVALID_USAGE:
        # Sequence[datetime] - Series[Any] (Timedelta) should work, see pandas-dev/pandas#62353
        _7 = a - left_td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]

    check(
        assert_type(left_ts.sub(s), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left_ts.sub(a), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    # Series[Any] (Timedelta).sub(datetime or Sequence[datetime]) fails at runtime,
    # which cannot be revealed by our static type checking
    # left_td.sub(s)
    # left_td.sub(a)

    check(assert_type(left_ts.rsub(s), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.rsub(a), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_td.rsub(s), pd.Series), pd.Series, pd.Timestamp)
    check(assert_type(left_td.rsub(a), pd.Series), pd.Series, pd.Timestamp)


def test_sub_ts_numpy_datetime() -> None:
    """Test pd.Series[Any] (Timestamp) - numpy datetime(s)"""
    s = np.datetime64(anchor)
    a = np.array([s + np.timedelta64(m, "m") for m in range(3)], np.datetime64)

    check(assert_type(left_ts - s, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left_ts - a, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    # Series[Any] (Timedelta) - np.datetime64 or np.typing.NDArray[np.datetime64]
    # fails at runtime,
    # which cannot be revealed by our static type checking
    # left_td - s
    # left_td - a

    # `numpy` typing gives the corresponding `ndarray`s in the static type
    # checking, where our `__rsub__` cannot override. At runtime, they return
    # `Series`.
    # microsoft/pyright#10924
    check(assert_type(s - left_ts, pd.Series), pd.Series, pd.Timedelta)
    check(
        assert_type(a - left_ts, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
        pd.Timedelta,
    )
    check(assert_type(s - left_td, pd.Series), pd.Series, pd.Timestamp)
    check(
        assert_type(a - left_td, Any),  # pyright: ignore[reportAssertTypeFailure]
        pd.Series,
        pd.Timestamp,
    )

    check(
        assert_type(left_ts.sub(s), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left_ts.sub(a), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    # Series[Any] (Timedelta).sub(np.datetime64 or np.typing.NDArray[np.datetime64])
    # fails at runtime,
    # which cannot be revealed by our static type checking
    # left_td.sub(s)
    # left_td.sub(a)

    check(assert_type(left_ts.rsub(s), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.rsub(a), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_td.rsub(s), pd.Series), pd.Series, pd.Timestamp)
    check(assert_type(left_td.rsub(a), pd.Series), pd.Series, pd.Timestamp)


def test_sub_ts_pd_datetime() -> None:
    """Test pd.Series[Any] (Timestamp) - Pandas datetime(s)"""
    s = pd.Timestamp(anchor)
    a = pd.Series([s + pd.Timedelta(minutes=m) for m in range(3)])

    check(assert_type(left_ts - s, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    check(assert_type(left_ts - a, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    # Series[Any] (Timedelta) - Timestamp or Series[Timestamp]
    # fails at runtime,
    # which cannot be revealed by our static type checking
    # left_td - s
    # left_td - a

    check(assert_type(s - left_ts, pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(a - left_ts, pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(s - left_td, pd.Series), pd.Series, pd.Timestamp)
    check(assert_type(a - left_td, pd.Series), pd.Series, pd.Timestamp)

    check(
        assert_type(left_ts.sub(s), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(left_ts.sub(a), "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    # Series[Any] (Timedelta).sub(Timestamp or Series[Timestamp])
    # fails at runtime,
    # which cannot be revealed by our static type checking
    # left_td.sub(s)
    # left_td.sub(a)

    check(assert_type(left_ts.rsub(s), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_ts.rsub(a), pd.Series), pd.Series, pd.Timedelta)
    check(assert_type(left_td.rsub(s), pd.Series), pd.Series, pd.Timestamp)
    check(assert_type(left_td.rsub(a), pd.Series), pd.Series, pd.Timestamp)


def test_sub_str_py_str() -> None:
    """Test pd.Series[Any] (int) - Python str"""
    s = "abc"

    if TYPE_CHECKING_INVALID_USAGE:
        _0 = left_i - s  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
        _1 = s - left_i  # type: ignore[operator] # pyright:ignore[reportOperatorIssue]
        left_i.sub(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
        left_i.rsub(s)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType,reportCallIssue]
