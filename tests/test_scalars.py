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
from typing_extensions import (
    TypeAlias,
    assert_type,
)

from pandas._libs.tslibs import (
    BaseOffset,
    NaTType,
)

from tests import check

from pandas.tseries.offsets import Day

if TYPE_CHECKING:
    from pandas.core.series import (
        OffsetSeries,
        PeriodSeries,
        TimedeltaSeries,
        TimestampSeries,
    )

    from pandas._typing import np_ndarray_bool
else:

    np_ndarray_bool = npt.NDArray[np.bool_]
    TimedeltaSeries = pd.Series
    TimestampSeries = pd.Series
    PeriodSeries: TypeAlias = pd.Series
    OffsetSeries: TypeAlias = pd.Series


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


def test_timestamp_types_init() -> None:
    check(assert_type(pd.Timestamp("2021-03-01T12"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timestamp(dt.date(2021, 3, 15)), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.Timestamp(dt.datetime(2021, 3, 10, 12)), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(pd.Timestamp(pd.Timestamp("2021-03-01T12")), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(pd.Timestamp(1515590000.1, unit="s"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(
            pd.Timestamp(1515590000.1, unit="s", tz="US/Pacific"), pd.Timestamp
        ),
        pd.Timestamp,
    )
    check(
        assert_type(pd.Timestamp(1515590000100000000), pd.Timestamp), pd.Timestamp
    )  # plain integer (nanosecond)
    check(assert_type(pd.Timestamp(2021, 3, 10, 12), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.Timestamp(year=2021, month=3, day=10, hour=12), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Timestamp(year=2021, month=3, day=10, hour=12, tz="US/Pacific"),
            pd.Timestamp,
        ),
        pd.Timestamp,
    )


def test_timestamp_types_arithmetic() -> None:
    ts: pd.Timestamp = pd.to_datetime("2021-03-01")
    ts2: pd.Timestamp = pd.to_datetime("2021-01-01")
    delta: pd.Timedelta = pd.to_timedelta("1 day")

    check(assert_type(ts - ts2, pd.Timedelta), pd.Timedelta)
    check(assert_type(ts + delta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - delta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - dt.datetime(2021, 1, 3), pd.Timedelta), pd.Timedelta)


def test_timestamp_types_comparison() -> None:
    ts: pd.Timestamp = pd.to_datetime("2021-03-01")
    ts2: pd.Timestamp = pd.to_datetime("2021-01-01")

    check(assert_type(ts < ts2, bool), bool)
    check(assert_type(ts > ts2, bool), bool)


def test_types_timestamp_series_comparisons() -> None:
    # GH 27
    df = pd.DataFrame(["2020-01-01", "2019-01-01"])
    tss = pd.to_datetime(df[0], format="%Y-%m-%d")
    ts = pd.to_datetime("2019-02-01", format="%Y-%m-%d")
    tssr = tss <= ts
    tssr2 = tss >= ts
    tssr3 = tss == ts
    check(assert_type(tssr, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(tssr2, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(tssr3, "pd.Series[bool]"), pd.Series, bool)
    # GH 265
    data = pd.date_range("2022-01-01", "2022-01-31", freq="D")
    s = pd.Series(data)
    ts2 = pd.Timestamp("2022-01-15")
    check(assert_type(s, TimestampSeries), pd.Series, pd.Timestamp)
    check(assert_type(ts2 <= s, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts2 >= s, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts2 < s, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts2 > s, "pd.Series[bool]"), pd.Series, bool)


def test_timestamp_types_pydatetime() -> None:
    ts: pd.Timestamp = pd.Timestamp("2021-03-01T12")
    check(assert_type(ts.to_pydatetime(), dt.datetime), dt.datetime)
    check(assert_type(ts.to_pydatetime(False), dt.datetime), dt.datetime)
    check(assert_type(ts.to_pydatetime(warn=True), dt.datetime), dt.datetime)


def test_timestamp_dateoffset_arithmetic() -> None:
    ts = pd.Timestamp("2022-03-18")
    do = pd.DateOffset(days=366)
    check(assert_type(ts + do, pd.Timestamp), pd.Timestamp)


def test_todatetime_fromnumpy() -> None:
    # GH 72
    t1 = np.datetime64("2022-07-04 02:30")
    check(assert_type(pd.to_datetime(t1), pd.Timestamp), pd.Timestamp)


def test_period_construction() -> None:
    p = pd.Period("2012-1-1", freq="D")
    check(assert_type(p, pd.Period), pd.Period)
    check(assert_type(pd.Period(p), pd.Period), pd.Period)
    check(assert_type(pd.Period("2012-1-1", freq=Day()), pd.Period), pd.Period)
    check(
        assert_type(pd.Period(freq="D", year=2012, day=1, month=1), pd.Period),
        pd.Period,
    )
    check(
        assert_type(pd.Period(None, "D", year=2012, day=1, month=1), pd.Period),
        pd.Period,
    )
    check(
        assert_type(pd.Period(None, "D", 1, year=2012, day=1, month=1), pd.Period),
        pd.Period,
    )
    check(
        assert_type(
            pd.Period(
                freq="s", year=2012, month=1, day=1, hour=12, minute=30, second=45
            ),
            pd.Period,
        ),
        pd.Period,
    )
    check(assert_type(pd.Period(freq="Q", year=2012, quarter=2), pd.Period), pd.Period)


def test_period_properties() -> None:
    p = pd.Period("2012-1-1", freq="D")

    check(assert_type(p.day, int), int)
    check(assert_type(p.day_of_week, int), int)
    check(assert_type(p.day_of_year, int), int)
    check(assert_type(p.dayofweek, int), int)
    check(assert_type(p.dayofyear, int), int)
    check(assert_type(p.days_in_month, int), int)
    check(assert_type(p.daysinmonth, int), int)
    check(assert_type(p.end_time, pd.Timestamp), pd.Timestamp)
    check(assert_type(p.freqstr, str), str)
    check(assert_type(p.hour, int), int)
    check(assert_type(p.is_leap_year, bool), bool)
    check(assert_type(p.minute, int), int)
    check(assert_type(p.month, int), int)
    check(assert_type(p.quarter, int), int)
    check(assert_type(p.qyear, int), int)
    check(assert_type(p.second, int), int)
    check(assert_type(p.start_time, pd.Timestamp), pd.Timestamp)
    check(assert_type(p.week, int), int)
    check(assert_type(p.weekday, int), int)
    check(assert_type(p.weekofyear, int), int)
    check(assert_type(p.year, int), int)
    check(assert_type(p.freq, BaseOffset), Day)
    check(assert_type(p.ordinal, int), int)

    p2 = pd.Period("2012-1-1", freq="2D")
    check(assert_type(p2.freq, BaseOffset), Day)


def test_period_add_subtract() -> None:
    p = pd.Period("2012-1-1", freq="D")

    as_pd_td = pd.Timedelta(1, "D")
    as_dt_td = dt.timedelta(days=1)
    as_np_td = np.timedelta64(1, "D")
    as_np_i64 = np.int64(1)
    as_int = int(1)
    as_period_index = pd.period_range("2012-1-1", periods=10, freq="D")
    check(assert_type(as_period_index, pd.PeriodIndex), pd.PeriodIndex)
    as_period = pd.Period("2012-1-1", freq="D")
    scale = 24 * 60 * 60 * 10**9
    as_td_series = pd.Series(pd.timedelta_range(scale, scale, freq="D"))
    check(assert_type(as_td_series, TimedeltaSeries), pd.Series)
    as_period_series = pd.Series(as_period_index)
    check(assert_type(as_period_series, PeriodSeries), pd.Series)
    as_timedelta_idx = pd.timedelta_range(scale, scale, freq="D")
    as_nat = pd.NaT

    check(assert_type(p + as_pd_td, pd.Period), pd.Period)
    check(assert_type(p + as_dt_td, pd.Period), pd.Period)
    check(assert_type(p + as_np_td, pd.Period), pd.Period)
    check(assert_type(p + as_np_i64, pd.Period), pd.Period)
    check(assert_type(p + as_int, pd.Period), pd.Period)
    check(assert_type(p + p.freq, pd.Period), pd.Period)
    # offset_index is tested below
    offset_index = p - as_period_index
    check(assert_type(p + offset_index, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p + as_td_series, PeriodSeries), pd.Series, pd.Period)
    check(assert_type(p + as_timedelta_idx, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p + as_nat, NaTType), NaTType)
    offset_series = as_period_series - as_period_series
    check(assert_type(offset_series, OffsetSeries), pd.Series)
    check(assert_type(p + offset_series, PeriodSeries), pd.Series, pd.Period)
    check(assert_type(p - as_pd_td, pd.Period), pd.Period)
    check(assert_type(p - as_dt_td, pd.Period), pd.Period)
    check(assert_type(p - as_np_td, pd.Period), pd.Period)
    check(assert_type(p - as_np_i64, pd.Period), pd.Period)
    check(assert_type(p - as_int, pd.Period), pd.Period)
    check(assert_type(offset_index, pd.Index), pd.Index)
    check(assert_type(p - as_period, BaseOffset), Day)
    check(assert_type(p - as_td_series, PeriodSeries), pd.Series, pd.Period)
    check(assert_type(p - as_timedelta_idx, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p - as_nat, NaTType), NaTType)
    check(assert_type(p - p.freq, pd.Period), pd.Period)

    # The __radd__ and __rsub__ methods are included to
    # establish the location of the concrete implementation
    # Those missing are using the __add__ of the other class
    check(assert_type(as_pd_td + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_pd_td), pd.Period), pd.Period)

    check(assert_type(as_dt_td + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_dt_td), pd.Period), pd.Period)

    check(assert_type(as_np_td + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_np_td), pd.Period), pd.Period)

    check(assert_type(as_np_i64 + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_np_i64), pd.Period), pd.Period)

    check(assert_type(as_int + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_int), pd.Period), pd.Period)

    check(assert_type(as_td_series + p, PeriodSeries), pd.Series, pd.Period)

    check(assert_type(as_timedelta_idx + p, pd.PeriodIndex), pd.PeriodIndex)

    check(assert_type(as_nat + p, NaTType), NaTType)
    check(assert_type(p.__radd__(as_nat), NaTType), NaTType)

    check(assert_type(p.freq + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(p.freq), pd.Period), pd.Period)

    check(assert_type(as_period_index - p, pd.Index), pd.Index)


def test_period_cmp() -> None:
    p = pd.Period("2012-1-1", freq="D")

    c_period = pd.Period("2012-1-1", freq="D")
    c_period_index = pd.period_range("2012-1-1", periods=10, freq="D")
    c_period_series = pd.Series(c_period_index)

    eq = check(assert_type(p == c_period, bool), bool)
    ne = check(assert_type(p != c_period, bool), bool)
    assert eq != ne

    eq_a = check(
        assert_type(p == c_period_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_q = check(
        assert_type(p != c_period_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_a != ne_q).all()

    eq_s = check(assert_type(p == c_period_series, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(p != c_period_series, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    eq = check(assert_type(c_period == p, bool), bool)
    ne = check(assert_type(c_period != p, bool), bool)
    assert eq != ne

    eq_a = check(
        assert_type(c_period_index == p, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_a = check(
        assert_type(c_period_index != p, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_a != ne_a).all()

    eq_s = check(assert_type(c_period_series == p, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(c_period_series != p, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    gt = check(assert_type(p > c_period, bool), bool)
    le = check(assert_type(p <= c_period, bool), bool)
    assert gt != le

    gt_a = check(assert_type(p > c_period_index, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(
        assert_type(p <= c_period_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_s = check(assert_type(p > c_period_series, "pd.Series[bool]"), pd.Series, bool)
    le_s = check(assert_type(p <= c_period_series, "pd.Series[bool]"), pd.Series, bool)
    assert (gt_s != le_s).all()

    gt = check(assert_type(c_period > p, bool), bool)
    le = check(assert_type(c_period <= p, bool), bool)
    assert gt != le

    gt_a = check(assert_type(c_period_index > p, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(
        assert_type(c_period_index <= p, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_s = check(assert_type(c_period_series > p, "pd.Series[bool]"), pd.Series, bool)
    le_s = check(assert_type(c_period_series <= p, "pd.Series[bool]"), pd.Series, bool)
    assert (gt_s != le_s).all()

    lt = check(assert_type(p < c_period, bool), bool)
    ge = check(assert_type(p >= c_period, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(p < c_period_index, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(
        assert_type(p >= c_period_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    lt_s = check(assert_type(p < c_period_series, "pd.Series[bool]"), pd.Series, bool)
    ge_s = check(assert_type(p >= c_period_series, "pd.Series[bool]"), pd.Series, bool)
    assert (lt_s != ge_s).all()

    lt = check(assert_type(c_period < p, bool), bool)
    ge = check(assert_type(c_period >= p, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(c_period_index < p, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(
        assert_type(c_period_index >= p, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    lt_s = check(assert_type(c_period_series < p, "pd.Series[bool]"), pd.Series, bool)
    ge_s = check(assert_type(c_period_series >= p, "pd.Series[bool]"), pd.Series, bool)
    assert (lt_s != ge_s).all()


def test_period_methods():
    p3 = pd.Period("2007-01", freq="M")
    check(assert_type(p3.to_timestamp("D", "S"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "E"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "start"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "end"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "Finish"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "Begin"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "End"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "s"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "e"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "finish"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "begin"), pd.Timestamp), pd.Timestamp)

    check(assert_type(p3.asfreq("D", "S"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "E"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "end"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "start"), pd.Period), pd.Period)

    check(assert_type(pd.Period.now("D"), pd.Period), pd.Period)
    check(assert_type(pd.Period.now(Day()), pd.Period), pd.Period)

    check(assert_type(p3.strftime("%Y-%m-%d"), str), str)
    check(assert_type(hash(p3), int), int)
