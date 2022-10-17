from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    cast,
)

import dateutil.tz
import numpy as np
from numpy import typing as npt
import pandas as pd
import pytz
from typing_extensions import assert_type

if TYPE_CHECKING:
    from pandas.core.series import TimedeltaSeries  # noqa: F401
    from pandas.core.series import TimestampSeries  # noqa: F401

    from pandas._typing import np_ndarray_bool
else:
    np_ndarray_bool = Any

from tests import check

from pandas.tseries.offsets import Day


def test_timestamp_construction() -> None:

    check(assert_type(pd.Timestamp("2000-1-1"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.Timestamp("2000-1-1", tz="US/Pacific"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Timestamp("2000-1-1", tz=pytz.timezone("US/Eastern")), pd.Timestamp
        ),
        pd.Timestamp,
    )
    check(
        assert_type(pd.Timestamp("2000-1-1", tz=dateutil.tz.UTC), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Timestamp(
                year=2000,
                month=1,
                day=1,
                hour=1,
                minute=1,
                second=1,
                microsecond=1,
                nanosecond=1,
            ),
            pd.Timestamp,
        ),
        pd.Timestamp,
    )
    check(assert_type(pd.Timestamp(1, unit="D"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp(1, unit="h"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp(1, unit="m"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp(1, unit="s"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp(1, unit="ms"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp(1, unit="us"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(
            pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27, fold=0),
            pd.Timestamp,
        ),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27, fold=1),
            pd.Timestamp,
        ),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Timestamp(
                year=2000,
                month=1,
                day=1,
                hour=1,
                minute=1,
                second=1,
                microsecond=1,
                nanosecond=1,
                tzinfo=dt.timezone(offset=dt.timedelta(hours=6), name="EST"),
            ),
            pd.Timestamp,
        ),
        pd.Timestamp,
    )


def test_timestamp_properties() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)

    check(assert_type(ts, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts.asm8, np.datetime64), np.datetime64)
    check(assert_type(ts.day_of_week, int), int)
    check(assert_type(ts.day_of_year, int), int)
    check(assert_type(ts.dayofweek, int), int)
    check(assert_type(ts.dayofyear, int), int)
    check(assert_type(ts.days_in_month, int), int)
    check(assert_type(ts.daysinmonth, int), int)
    check(assert_type(ts.is_leap_year, bool), bool)
    check(assert_type(ts.is_month_end, bool), bool)
    check(assert_type(ts.is_month_start, bool), bool)
    check(assert_type(ts.is_quarter_end, bool), bool)
    check(assert_type(ts.is_quarter_start, bool), bool)
    check(assert_type(ts.is_year_end, bool), bool)
    check(assert_type(ts.is_year_start, bool), bool)
    check(assert_type(ts.quarter, int), int)
    check(assert_type(ts.tz, Optional[dt.tzinfo]), type(None))
    check(assert_type(ts.week, int), int)
    check(assert_type(ts.weekofyear, int), int)
    check(assert_type(ts.day, int), int)
    check(assert_type(ts.fold, int), int)
    check(assert_type(ts.hour, int), int)
    check(assert_type(ts.microsecond, int), int)
    check(assert_type(ts.minute, int), int)
    check(assert_type(ts.month, int), int)
    check(assert_type(ts.nanosecond, int), int)
    check(assert_type(ts.second, int), int)
    check(assert_type(ts.tzinfo, Optional[dt.tzinfo]), type(None))
    check(assert_type(ts.value, int), int)
    check(assert_type(ts.year, int), int)


def test_timestamp_add_sub() -> None:
    ts = pd.Timestamp("2000-1-1")
    ts_series = cast("TimedeltaSeries", pd.Series([1, 2], dtype="timedelta64[ns]"))
    np_td64_arr: npt.NDArray[np.timedelta64] = np.array([1, 2], dtype="timedelta64[ns]")

    as1 = pd.Timedelta(days=1)
    as2 = dt.timedelta(days=1)
    as3 = 3 * Day()
    as4 = pd.TimedeltaIndex([1, 2, 3], "D")
    as5 = ts_series
    as6 = np_td64_arr

    check(assert_type(ts + as1, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts + as2, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts + as3, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts + as4, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ts + as5, "TimestampSeries"), pd.Series)
    check(assert_type(ts + as6, npt.NDArray[np.datetime64]), np.ndarray)

    check(assert_type(as1 + ts, pd.Timestamp), pd.Timestamp)
    check(assert_type(as2 + ts, pd.Timestamp), pd.Timestamp)
    check(assert_type(as3 + ts, pd.Timestamp), pd.Timestamp)
    check(assert_type(as4 + ts, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(as5 + ts, "TimestampSeries"), pd.Series)
    # pyright and mypy disagree on the type of this expression
    # pyright: Any, mypy: npt.NDArray[np.datetime64]
    check(
        assert_type(
            as6 + ts,  # pyright: ignore [reportGeneralTypeIssues]
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
    )

    check(assert_type(ts - as1, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as2, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as3, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as4, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ts - as5, "TimestampSeries"), pd.Series)
    check(assert_type(ts - as6, npt.NDArray[np.datetime64]), np.ndarray)


def test_timestamp_cmp() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)

    np_dt64_arr: npt.NDArray[np.datetime64] = np.array(
        [1, 2, 3], dtype="datetime64[ns]"
    )

    c1 = ts
    c2 = np.datetime64(1, "ns")
    c3 = dt.datetime(year=2000, month=1, day=1)
    c4 = pd.DatetimeIndex(["2000-1-1"])
    c5 = np_dt64_arr
    c6 = pd.Series([1, 2, 3], dtype="datetime64[ns]")

    # Use xor to ensure one is True and the other is False
    # Correctness ensures since tested to be bools
    gt = check(assert_type(ts > c1, bool), bool)
    lte = check(assert_type(ts <= c1, bool), bool)
    assert gt != lte

    gt = check(assert_type(ts > c2, bool), bool)
    lte = check(assert_type(ts <= c2, bool), bool)
    assert gt != lte

    gt = check(assert_type(ts > c3, bool), bool)
    lte = check(assert_type(ts <= c3, bool), bool)
    assert gt != lte

    check(assert_type(ts > c4, np_ndarray_bool), np.ndarray)
    check(assert_type(ts <= c4, np_ndarray_bool), np.ndarray)

    check(assert_type(ts > c5, np_ndarray_bool), np.ndarray)
    check(assert_type(ts <= c5, np_ndarray_bool), np.ndarray)

    check(assert_type(ts > c6, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts <= c6, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(c2 > ts, Any), np.bool_)
    check(assert_type(c2 <= ts, Any), np.bool_)

    gt = check(assert_type(c3 > ts, bool), bool)
    lte = check(assert_type(c3 <= ts, bool), bool)
    assert gt != lte

    check(assert_type(c4 > ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c4 <= ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c5 > ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c5 <= ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c6 > ts, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(c6 <= ts, "pd.Series[bool]"), pd.Series, bool)

    gte = check(assert_type(ts >= c1, bool), bool)
    lt = check(assert_type(ts < c1, bool), bool)
    assert gte != lt

    gte = check(assert_type(ts >= c2, bool), bool)
    lt = check(assert_type(ts < c2, bool), bool)
    assert gte != lt

    gte = check(assert_type(ts >= c3, bool), bool)
    lt = check(assert_type(ts < c3, bool), bool)
    assert gte != lt

    check(assert_type(ts >= c4, np_ndarray_bool), np.ndarray)
    check(assert_type(ts < c4, np_ndarray_bool), np.ndarray)

    check(assert_type(ts >= c5, np_ndarray_bool), np.ndarray)
    check(assert_type(ts < c5, np_ndarray_bool), np.ndarray)

    check(assert_type(ts >= c6, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts < c6, "pd.Series[bool]"), pd.Series, bool)

    gte = check(assert_type(c3 >= ts, bool), bool)
    lt = check(assert_type(c3 < ts, bool), bool)
    assert gte != lt

    check(assert_type(c2 >= ts, Any), np.bool_)
    check(assert_type(c2 < ts, Any), np.bool_)

    check(assert_type(c4 >= ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c4 < ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c5 >= ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c5 < ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c6 >= ts, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(c6 < ts, "pd.Series[bool]"), pd.Series, bool)

    eq = check(assert_type(ts == c1, bool), bool)
    ne = check(assert_type(ts != c1, bool), bool)
    assert eq != ne

    eq = check(assert_type(ts == c2, bool), bool)
    ne = check(assert_type(ts != c2, bool), bool)
    assert eq != ne

    eq = check(assert_type(ts == c3, bool), bool)
    ne = check(assert_type(ts != c3, bool), bool)
    assert eq != ne

    check(assert_type(ts == c4, np_ndarray_bool), np.ndarray)
    check(assert_type(ts != c4, np_ndarray_bool), np.ndarray)

    check(assert_type(ts == c5, np_ndarray_bool), np.ndarray)
    check(assert_type(ts != c5, np_ndarray_bool), np.ndarray)

    check(assert_type(ts == c6, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts != c6, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(c2 == ts, Any), np.bool_)
    check(assert_type(c2 != ts, Any), np.bool_)

    eq = check(assert_type(c3 == ts, bool), bool)
    ne = check(assert_type(c3 != ts, bool), bool)
    assert eq != ne

    check(assert_type(c4 == ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c4 != ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c5 != ts, Any), np.ndarray)
    check(assert_type(c5 == ts, Any), np.ndarray)

    check(assert_type(c6 == ts, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(c6 != ts, "pd.Series[bool]"), pd.Series, bool)
