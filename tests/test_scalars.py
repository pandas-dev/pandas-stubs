from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)

import dateutil.tz
import numpy as np
from numpy import typing as npt
import pandas as pd
import pytz
from typing_extensions import assert_type

if TYPE_CHECKING:
    from pandas.core.series import (
        TimedeltaSeries,
        TimestampSeries,
    )

    from pandas._typing import np_ndarray_bool
else:
    np_ndarray_bool = npt.NDArray[np.bool_]
    TimedeltaSeries = pd.Series
    TimestampSeries = pd.Series


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
    np_td64_arr: npt.NDArray[np.timedelta64] = np.array([1, 2], dtype="timedelta64[ns]")

    as_pd_timedelta = pd.Timedelta(days=1)
    as_dt_timedelta = dt.timedelta(days=1)
    as_offset = 3 * Day()
    as_timedelta_index = pd.TimedeltaIndex([1, 2, 3], "D")
    as_timedelta_series = pd.Series(as_timedelta_index)
    check(assert_type(as_timedelta_series, TimedeltaSeries), pd.Series)
    as_np_ndarray_td64 = np_td64_arr

    check(assert_type(ts + as_pd_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_pd_timedelta + ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts + as_dt_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_dt_timedelta + ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts + as_offset, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_offset + ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts + as_timedelta_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(as_timedelta_index + ts, pd.DatetimeIndex), pd.DatetimeIndex)

    check(assert_type(ts + as_timedelta_series, "TimestampSeries"), pd.Series)
    check(assert_type(as_timedelta_series + ts, "TimestampSeries"), pd.Series)

    check(assert_type(ts + as_np_ndarray_td64, npt.NDArray[np.datetime64]), np.ndarray)
    # pyright and mypy disagree on the type of this expression
    # pyright: Any, mypy: npt.NDArray[np.datetime64]
    check(
        assert_type(
            as_np_ndarray_td64 + ts,  # pyright: ignore [reportGeneralTypeIssues]
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
    )

    # Reverse order is not possible for all of these
    check(assert_type(ts - as_pd_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_dt_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_offset, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_timedelta_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ts - as_timedelta_series, "TimestampSeries"), pd.Series)
    check(assert_type(ts - as_np_ndarray_td64, npt.NDArray[np.datetime64]), np.ndarray)


def test_timestamp_cmp() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)

    np_dt64_arr: npt.NDArray[np.datetime64] = np.array(
        [1, 2, 3], dtype="datetime64[ns]"
    )

    c_timestamp = ts
    c_np_dt64 = np.datetime64(1, "ns")
    c_dt_datetime = dt.datetime(year=2000, month=1, day=1)
    c_datetimeindex = pd.DatetimeIndex(["2000-1-1"])
    c_np_ndarray_dt64 = np_dt64_arr
    c_series_dt64 = pd.Series([1, 2, 3], dtype="datetime64[ns]")
    c_series_timestamp = pd.Series(pd.DatetimeIndex(["2000-1-1"]))
    check(assert_type(c_series_timestamp, TimestampSeries), pd.Series)
    # Use xor to ensure one is True and the other is False
    # Correctness ensures since tested to be bools
    gt = check(assert_type(ts > c_timestamp, bool), bool)
    lte = check(assert_type(ts <= c_timestamp, bool), bool)
    assert gt != lte

    gt = check(assert_type(ts > c_np_dt64, bool), bool)
    lte = check(assert_type(ts <= c_np_dt64, bool), bool)
    assert gt != lte

    gt = check(assert_type(ts > c_dt_datetime, bool), bool)
    lte = check(assert_type(ts <= c_dt_datetime, bool), bool)
    assert gt != lte

    check(assert_type(ts > c_datetimeindex, np_ndarray_bool), np.ndarray)
    check(assert_type(ts <= c_datetimeindex, np_ndarray_bool), np.ndarray)

    check(assert_type(ts > c_np_ndarray_dt64, np_ndarray_bool), np.ndarray)
    check(assert_type(ts <= c_np_ndarray_dt64, np_ndarray_bool), np.ndarray)

    check(assert_type(ts > c_series_timestamp, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts <= c_series_timestamp, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(ts > c_series_dt64, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts <= c_series_dt64, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(c_np_dt64 > ts, Any), np.bool_)
    check(assert_type(c_np_dt64 <= ts, Any), np.bool_)

    gt = check(assert_type(c_dt_datetime > ts, bool), bool)
    lte = check(assert_type(c_dt_datetime <= ts, bool), bool)
    assert gt != lte

    check(assert_type(c_datetimeindex > ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c_datetimeindex <= ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c_np_ndarray_dt64 > ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c_np_ndarray_dt64 <= ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c_series_dt64 > ts, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(c_series_dt64 <= ts, "pd.Series[bool]"), pd.Series, bool)

    gte = check(assert_type(ts >= c_timestamp, bool), bool)
    lt = check(assert_type(ts < c_timestamp, bool), bool)
    assert gte != lt

    gte = check(assert_type(ts >= c_np_dt64, bool), bool)
    lt = check(assert_type(ts < c_np_dt64, bool), bool)
    assert gte != lt

    gte = check(assert_type(ts >= c_dt_datetime, bool), bool)
    lt = check(assert_type(ts < c_dt_datetime, bool), bool)
    assert gte != lt

    check(assert_type(ts >= c_datetimeindex, np_ndarray_bool), np.ndarray)
    check(assert_type(ts < c_datetimeindex, np_ndarray_bool), np.ndarray)

    check(assert_type(ts >= c_np_ndarray_dt64, np_ndarray_bool), np.ndarray)
    check(assert_type(ts < c_np_ndarray_dt64, np_ndarray_bool), np.ndarray)

    check(assert_type(ts >= c_series_timestamp, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts < c_series_timestamp, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(ts >= c_series_dt64, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts < c_series_dt64, "pd.Series[bool]"), pd.Series, bool)

    gte = check(assert_type(c_dt_datetime >= ts, bool), bool)
    lt = check(assert_type(c_dt_datetime < ts, bool), bool)
    assert gte != lt

    check(assert_type(c_np_dt64 >= ts, Any), np.bool_)
    check(assert_type(c_np_dt64 < ts, Any), np.bool_)

    check(assert_type(c_datetimeindex >= ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c_datetimeindex < ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c_np_ndarray_dt64 >= ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c_np_ndarray_dt64 < ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c_series_dt64 >= ts, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(c_series_dt64 < ts, "pd.Series[bool]"), pd.Series, bool)

    eq = check(assert_type(ts == c_timestamp, bool), bool)
    ne = check(assert_type(ts != c_timestamp, bool), bool)
    assert eq != ne

    eq = check(assert_type(ts == c_np_dt64, bool), bool)
    ne = check(assert_type(ts != c_np_dt64, bool), bool)
    assert eq != ne

    eq = check(assert_type(ts == c_dt_datetime, bool), bool)
    ne = check(assert_type(ts != c_dt_datetime, bool), bool)
    assert eq != ne

    check(assert_type(ts == c_datetimeindex, np_ndarray_bool), np.ndarray)
    check(assert_type(ts != c_datetimeindex, np_ndarray_bool), np.ndarray)

    check(assert_type(ts == c_np_ndarray_dt64, np_ndarray_bool), np.ndarray)
    check(assert_type(ts != c_np_ndarray_dt64, np_ndarray_bool), np.ndarray)

    check(assert_type(ts == c_series_timestamp, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts != c_series_timestamp, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(ts == c_series_dt64, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts != c_series_dt64, "pd.Series[bool]"), pd.Series, bool)

    check(assert_type(c_np_dt64 == ts, Any), np.bool_)
    check(assert_type(c_np_dt64 != ts, Any), np.bool_)

    eq = check(assert_type(c_dt_datetime == ts, bool), bool)
    ne = check(assert_type(c_dt_datetime != ts, bool), bool)
    assert eq != ne

    check(assert_type(c_datetimeindex == ts, np_ndarray_bool), np.ndarray)
    check(assert_type(c_datetimeindex != ts, np_ndarray_bool), np.ndarray)

    check(assert_type(c_np_ndarray_dt64 != ts, Any), np.ndarray)
    check(assert_type(c_np_ndarray_dt64 == ts, Any), np.ndarray)

    check(assert_type(c_series_dt64 == ts, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(c_series_dt64 != ts, "pd.Series[bool]"), pd.Series, bool)
