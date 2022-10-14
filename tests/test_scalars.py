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
import pytest
import pytz
from typing_extensions import assert_type

from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.timedeltas import Components

if TYPE_CHECKING:
    from pandas.core.series import PeriodSeries  # noqa: F401
    from pandas.core.series import TimedeltaSeries  # noqa: F401
    from pandas.core.series import TimestampSeries  # noqa: F401

    from pandas._typing import np_ndarray_bool
else:
    np_ndarray_bool = Any

from tests import check

from pandas.tseries.offsets import Day


def test_period() -> None:
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

    as0 = pd.Timedelta(1, "D")
    as1 = dt.timedelta(days=1)
    as2 = np.timedelta64(1, "D")
    as3 = np.int64(1)
    as4 = int(1)
    as5 = pd.period_range("2012-1-1", periods=10, freq="D")
    as6 = pd.Period("2012-1-1", freq="D")
    as7 = cast("TimedeltaSeries", pd.Series([pd.Timedelta(days=1)]))
    as8 = cast("PeriodSeries", pd.Series([as6]))

    check(assert_type(p + as0, pd.Period), pd.Period)
    check(assert_type(p + as1, pd.Period), pd.Period)
    check(assert_type(p + as2, pd.Period), pd.Period)
    check(assert_type(p + as3, pd.Period), pd.Period)
    check(assert_type(p + as4, pd.Period), pd.Period)
    check(assert_type(p + p.freq, pd.Period), pd.Period)
    check(assert_type(p + (p - as5), pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p + as7, "PeriodSeries"), pd.Series)
    das8 = cast("TimedeltaSeries", (as8 - as8))
    check(assert_type(p + das8, "PeriodSeries"), pd.Series)
    check(assert_type(p - as0, pd.Period), pd.Period)
    check(assert_type(p - as1, pd.Period), pd.Period)
    check(assert_type(p - as2, pd.Period), pd.Period)
    check(assert_type(p - as3, pd.Period), pd.Period)
    check(assert_type(p - as4, pd.Period), pd.Period)
    check(assert_type(p - as5, pd.Index), pd.Index)
    check(assert_type(p - as6, BaseOffset), Day)
    check(assert_type(p - as7, "PeriodSeries"), pd.Series)
    check(assert_type(p - p.freq, pd.Period), pd.Period)

    check(assert_type(as0 + p, pd.Period), pd.Period)
    check(assert_type(as1 + p, pd.Period), pd.Period)
    check(assert_type(as2 + p, pd.Period), pd.Period)
    check(assert_type(as3 + p, pd.Period), pd.Period)
    check(assert_type(as4 + p, pd.Period), pd.Period)
    check(assert_type(as7 + p, "PeriodSeries"), pd.Series)
    check(assert_type(p.freq + p, pd.Period), pd.Period)
    # TODO: PeriodIndex should have a __sub__ with correct types, this op is valid
    #  and so the assert_type is skipped
    check(as5 - p, pd.Index)  # type: ignore[operator]

    check(assert_type(p.__radd__(as0), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as1), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as2), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as3), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as4), pd.Period), pd.Period)
    check(assert_type(p.__radd__(p.freq), pd.Period), pd.Period)

    c0 = pd.Period("2012-1-1", freq="D")
    c1 = pd.period_range("2012-1-1", periods=10, freq="D")

    check(assert_type(p == c0, bool), bool)
    check(assert_type(p == c1, np_ndarray_bool), np.ndarray)
    check(assert_type(c0 == p, bool), bool)
    check(assert_type(c1 == p, np_ndarray_bool), np.ndarray)

    check(assert_type(p != c0, bool), bool)
    check(assert_type(p != c1, np_ndarray_bool), np.ndarray)
    check(assert_type(c0 != p, bool), bool)
    check(assert_type(c1 != p, np_ndarray_bool), np.ndarray)

    check(assert_type(p > c0, bool), bool)
    check(assert_type(p > c1, np_ndarray_bool), np.ndarray)
    check(assert_type(c0 > p, bool), bool)
    check(assert_type(c1 > p, np_ndarray_bool), np.ndarray)

    check(assert_type(p < c0, bool), bool)
    check(assert_type(p < c1, np_ndarray_bool), np.ndarray)
    check(assert_type(c0 < p, bool), bool)
    check(assert_type(c1 < p, np_ndarray_bool), np.ndarray)

    check(assert_type(p <= c0, bool), bool)
    check(assert_type(p <= c1, np_ndarray_bool), np.ndarray)
    check(assert_type(c0 <= p, bool), bool)
    check(assert_type(c1 <= p, np_ndarray_bool), np.ndarray)

    check(assert_type(p >= c0, bool), bool)
    check(assert_type(p >= c1, np_ndarray_bool), np.ndarray)
    check(assert_type(c0 >= p, bool), bool)
    check(assert_type(c1 >= p, np_ndarray_bool), np.ndarray)

    p3 = pd.Period("2007-01", freq="M")
    check(assert_type(p3.to_timestamp("D", "S"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "E"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "Finish"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "End"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "Begin"), pd.Timestamp), pd.Timestamp)
    check(assert_type(p3.to_timestamp("D", "Start"), pd.Timestamp), pd.Timestamp)

    check(assert_type(p3.asfreq("D", "S"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "E"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "Finish"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "Begin"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "Start"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "End"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "end"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "start"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "begin"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "finish"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "s"), pd.Period), pd.Period)
    check(assert_type(p3.asfreq(Day(), "e"), pd.Period), pd.Period)

    check(assert_type(pd.Period.now("D"), pd.Period), pd.Period)
    check(assert_type(pd.Period.now(Day()), pd.Period), pd.Period)

    check(assert_type(p.strftime("%Y-%m-%d"), str), str)
    check(assert_type(hash(p), int), int)


def test_timedelta() -> None:
    check(assert_type(pd.Timedelta(1, "W"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "w"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "D"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "d"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "days"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "day"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "hours"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "hour"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "hr"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "m"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "minute"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "min"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "minutes"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "t"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "seconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "sec"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "second"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "ms"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "milliseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "millisecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "milli"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "millis"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "l"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "us"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "microseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "microsecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "µs"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "micro"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "micros"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "u"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "ns"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nanoseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nano"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nanos"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nanosecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "n"), pd.Timedelta), pd.Timedelta)

    check(assert_type(pd.Timedelta("1 W"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 w"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 D"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 d"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 days"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 day"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 hours"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 hour"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 hr"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 m"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 minute"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 min"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 minutes"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 t"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 seconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 sec"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 second"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 ms"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 milliseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 millisecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 milli"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 millis"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 l"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 us"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 microseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 microsecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 µs"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 micro"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 micros"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 u"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 ns"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nanoseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nano"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nanos"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nanosecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 n"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(days=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(seconds=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(microseconds=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(minutes=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(hours=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(weeks=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(milliseconds=1), pd.Timedelta), pd.Timedelta)

    td = pd.Timedelta("1 day")
    check(assert_type(td.value, int), int)
    check(assert_type(td.asm8, np.timedelta64), np.timedelta64)

    check(assert_type(td.days, int), int)
    check(assert_type(td.microseconds, int), int)
    check(assert_type(td.nanoseconds, int), int)
    check(assert_type(td.seconds, int), int)
    check(assert_type(td.value, int), int)
    check(assert_type(td.resolution_string, str), str)
    check(assert_type(td.components, Components), Components)

    check(assert_type(td.ceil("D"), pd.Timedelta), pd.Timedelta)
    check(assert_type(td.floor(Day()), pd.Timedelta), pd.Timedelta)
    check(assert_type(td.isoformat(), str), str)
    check(assert_type(td.round("s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(td.to_numpy(), np.timedelta64), np.timedelta64)
    check(assert_type(td.to_pytimedelta(), dt.timedelta), dt.timedelta)
    check(assert_type(td.to_timedelta64(), np.timedelta64), np.timedelta64)
    check(assert_type(td.total_seconds(), float), float)
    check(assert_type(td.view(np.int64), object), np.int64)
    check(assert_type(td.view("i8"), object), np.int64)

    ndarray_td64: npt.NDArray[np.timedelta64] = np.array(
        [1, 2, 3], dtype="timedelta64[D]"
    )
    ndarray_dt64: npt.NDArray[np.datetime64] = np.array(
        [1, 2, 3], dtype="datetime64[D]"
    )

    check(assert_type(td + pd.Period("2012-01-01", freq="D"), pd.Period), pd.Period)
    check(assert_type(td + pd.Timestamp("2012-01-01"), pd.Timestamp), pd.Timestamp)
    check(assert_type(td + dt.datetime(2012, 1, 1), pd.Timestamp), pd.Timestamp)
    check(assert_type(td + dt.date(2012, 1, 1), dt.date), dt.date)
    check(assert_type(td + np.datetime64(1, "ns"), pd.Timestamp), pd.Timestamp)
    check(assert_type(td + dt.timedelta(days=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(td + np.timedelta64(1, "D"), pd.Timedelta), pd.Timedelta)
    check(
        assert_type(
            td + pd.period_range("2012-01-01", periods=3, freq="D"), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(td + pd.date_range("2012-01-01", periods=3), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            td + ndarray_td64,
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            td + ndarray_dt64,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
    )

    check(assert_type(td - td, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - dt.timedelta(days=1), pd.Timedelta), pd.Timedelta)
    check(assert_type(td - np.timedelta64(1, "D"), pd.Timedelta), pd.Timedelta)
    check(
        assert_type(
            td - ndarray_td64,
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
    )
    # pyright appears to get some things wrong when __rsub__ is called,
    # hence pyright ignores
    check(assert_type(pd.Period("2012-01-01", freq="D") - td, pd.Period), pd.Period)
    check(assert_type(pd.Timestamp("2012-01-01") - td, pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.datetime(2012, 1, 1) - td, dt.datetime), dt.datetime)
    check(assert_type(dt.date(2012, 1, 1) - td, dt.date), dt.date)
    check(assert_type(np.datetime64(1, "ns") - td, pd.Timestamp), pd.Timestamp)
    check(
        assert_type(
            dt.timedelta(days=1) - td,  # pyright: ignore[reportGeneralTypeIssues]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(assert_type(np.timedelta64(1, "D") - td, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(
            pd.period_range("2012-01-01", periods=3, freq="D") - td, pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(pd.date_range("2012-01-01", periods=3) - td, pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            ndarray_td64 - td,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
    )
    check(
        assert_type(
            ndarray_dt64 - td,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
    )

    with pytest.warns(FutureWarning):
        i_idx = cast(pd.Int64Index, pd.Index([1, 2, 3], dtype=int))
        f_idx = cast(pd.Float64Index, pd.Index([1.2, 2.2, 3.4], dtype=float))

    check(assert_type(td * 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * 3.5, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * np.array([1, 2, 3]), np.ndarray), np.ndarray)
    check(assert_type(td * np.array([1.2, 2.2, 3.4]), np.ndarray), np.ndarray)
    check(assert_type(td * pd.Series([1, 2, 3]), pd.Series), pd.Series)
    check(assert_type(td * pd.Series([1.2, 2.2, 3.4]), pd.Series), pd.Series)
    check(assert_type(td * i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td * f_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(3 * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(3.5 * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Series([1, 2, 3]) * td, "TimedeltaSeries"), pd.Series)
    check(assert_type(pd.Series([1.2, 2.2, 3.4]) * td, "TimedeltaSeries"), pd.Series)
    check(assert_type(i_idx * td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(f_idx * td, pd.TimedeltaIndex), pd.TimedeltaIndex)

    np_intp_arr: npt.NDArray[np.integer] = np.array([1, 2, 3])
    np_float_arr: npt.NDArray[np.floating] = np.array([1, 2, 3])
    check(assert_type(td // td, int), int)
    check(assert_type(td // 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // 3.5, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // np_intp_arr, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td // np_float_arr, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td // pd.Series([1, 2, 3]), pd.Series), pd.Series)
    check(assert_type(td // pd.Series([1.2, 2.2, 3.4]), pd.Series), pd.Series)
    check(assert_type(td // i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td // f_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    # Note: None of the reverse floordiv work
    # check(assert_type(3 // td, pd.Timedelta), pd.Timedelta)
    # check(assert_type(3.5// td, pd.Timedelta), pd.Timedelta)
    # check(assert_type(np_intp_arr// td, npt.NDArray[np.timedelta64]), np.ndarray)
    # check(assert_type(np_float_arr// td, npt.NDArray[np.timedelta64]), np.ndarray)
    # check(assert_type(pd.Series([1, 2, 3])// td, pd.Series), pd.Series)
    # check(assert_type(pd.Series([1.2, 2.2, 3.4])// td, pd.Series), pd.Series)
    # check(assert_type(i_idx, pd.TimedeltaIndex)// td, pd.TimedeltaIndex)
    # check( assert_type(f_idx// td, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(td / td, float), float)
    check(assert_type(td / 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td / 3.5, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td / np.array([1, 2, 3]), npt.NDArray[np.timedelta64]), np.ndarray
    )
    check(
        assert_type(td / np.array([1.2, 2.2, 3.4]), npt.NDArray[np.timedelta64]),
        np.ndarray,
    )
    check(assert_type(td / pd.Series([1, 2, 3]), pd.Series), pd.Series)
    check(assert_type(td / pd.Series([1.2, 2.2, 3.4]), pd.Series), pd.Series)
    check(assert_type(td / i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td / f_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    # Note: None of the reverse truediv work
    # check(assert_type(3 / td, pd.Timedelta), pd.Timedelta)
    # check(assert_type(3.5 / td, pd.Timedelta), pd.Timedelta)
    # check(assert_type(np.array([1, 2, 3]) / td, npt.NDArray[np.timedelta64]), np.ndarray)
    # check(assert_type(np.array([1.2, 2.2, 3.4]) / td, npt.NDArray[np.timedelta64]),np.ndarray,)
    # check(assert_type(pd.Series([1, 2, 3]) / td, pd.Series), pd.Series)
    # check(assert_type(pd.Series([1.2, 2.2, 3.4]) / td, pd.Series), pd.Series)
    # check(assert_type(i_idx / td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    # check(assert_type(f_idx / td, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(td % 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td % 3.5, pd.Timedelta), pd.Timedelta)
    check(assert_type(td % td, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td % np.array([1, 2, 3]), npt.NDArray[np.timedelta64]), np.ndarray
    )
    check(
        assert_type(td % np.array([1.2, 2.2, 3.4]), npt.NDArray[np.timedelta64]),
        np.ndarray,
    )
    check(assert_type(td % pd.Series([1, 2, 3]), pd.Series), pd.Series)
    check(assert_type(td % pd.Series([1.2, 2.2, 3.4]), pd.Series), pd.Series)
    check(assert_type(td % i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(
        assert_type(td % f_idx, pd.TimedeltaIndex),
        pd.TimedeltaIndex,
    )

    check(assert_type(td.__abs__(), pd.Timedelta), pd.Timedelta)
    check(assert_type(-td, pd.Timedelta), pd.Timedelta)
    check(assert_type(+td, pd.Timedelta), pd.Timedelta)

    check(assert_type(td < td, bool), bool)
    check(assert_type(td < dt.timedelta(days=1), bool), bool)
    check(assert_type(td < np.timedelta64(1, "D"), bool), bool)
    check(assert_type(td < ndarray_td64, np_ndarray_bool), np.ndarray)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") < td, np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(dt.timedelta(days=1) < td, bool), bool)
    check(assert_type(ndarray_td64 < td, np_ndarray_bool), np.ndarray)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") < td, np_ndarray_bool),
        np.ndarray,
    )

    check(assert_type(td > td, bool), bool)
    check(assert_type(td > dt.timedelta(days=1), bool), bool)
    check(assert_type(td > np.timedelta64(1, "D"), bool), bool)
    check(assert_type(td > ndarray_td64, np_ndarray_bool), np.ndarray)
    check(
        assert_type(td > pd.TimedeltaIndex([1, 2, 3], unit="D"), np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(dt.timedelta(days=1) > td, bool), bool)
    check(assert_type(ndarray_td64 > td, np_ndarray_bool), np.ndarray)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") > td, np_ndarray_bool),
        np.ndarray,
    )

    check(assert_type(td <= td, bool), bool)
    check(assert_type(td <= dt.timedelta(days=1), bool), bool)
    check(assert_type(td <= np.timedelta64(1, "D"), bool), bool)
    check(assert_type(td <= ndarray_td64, np_ndarray_bool), np.ndarray)
    check(
        assert_type(td <= pd.TimedeltaIndex([1, 2, 3], unit="D"), np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(dt.timedelta(days=1) <= td, bool), bool)
    check(assert_type(ndarray_td64 <= td, np_ndarray_bool), np.ndarray)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") <= td, np_ndarray_bool),
        np.ndarray,
    )

    check(assert_type(td >= td, bool), bool)
    check(assert_type(td >= dt.timedelta(days=1), bool), bool)
    check(assert_type(td >= np.timedelta64(1, "D"), bool), bool)
    check(assert_type(td >= ndarray_td64, np_ndarray_bool), np.ndarray)
    check(
        assert_type(td >= pd.TimedeltaIndex([1, 2, 3], unit="D"), np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(dt.timedelta(days=1) >= td, bool), bool)
    check(assert_type(ndarray_td64 >= td, np_ndarray_bool), np.ndarray)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") >= td, np_ndarray_bool),
        np.ndarray,
    )

    check(assert_type(td == td, bool), bool)
    check(assert_type(td == dt.timedelta(days=1), bool), bool)
    check(assert_type(td == np.timedelta64(1, "D"), bool), bool)
    check(assert_type(td == ndarray_td64, np_ndarray_bool), np.ndarray)
    check(
        assert_type(td == pd.TimedeltaIndex([1, 2, 3], unit="D"), np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(td == pd.Series([1, 2, 3]), pd.Series), pd.Series)
    check(assert_type(dt.timedelta(days=1) == td, bool), bool)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") == td, np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(pd.Series([1, 2, 3]) == td, "pd.Series[bool]"), pd.Series)

    check(assert_type(td == 1, bool), bool)
    check(assert_type(td == (3 + 2j), bool), bool)

    check(assert_type(td != td, bool), bool)
    check(assert_type(td != dt.timedelta(days=1), bool), bool)
    check(assert_type(td != np.timedelta64(1, "D"), bool), bool)
    check(assert_type(td != ndarray_td64, np_ndarray_bool), np.ndarray)
    check(
        assert_type(td != pd.TimedeltaIndex([1, 2, 3], unit="D"), np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(td != pd.Series([1, 2, 3]), pd.Series), pd.Series)
    check(assert_type(dt.timedelta(days=1) != td, bool), bool)
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], unit="D") != td, np_ndarray_bool),
        np.ndarray,
    )
    check(assert_type(pd.Series([1, 2, 3]) != td, "pd.Series[bool]"), pd.Series)
    check(assert_type(td != 1, bool), bool)
    check(assert_type(td != (3 + 2j), bool), bool)

    # Mismatch due to NumPy ops returning Any
    check(assert_type(np.array([1, 2, 3]) * td, Any), np.ndarray)
    check(assert_type(np.array([1.2, 2.2, 3.4]) * td, Any), np.ndarray)
    check(assert_type(np.timedelta64(1, "D") < td, Any), np.bool_)
    check(assert_type(np.timedelta64(1, "D") > td, Any), np.bool_)
    check(assert_type(np.timedelta64(1, "D") <= td, Any), np.bool_)
    check(assert_type(np.timedelta64(1, "D") >= td, Any), np.bool_)
    check(assert_type(np.timedelta64(1, "D") == td, Any), np.bool_)
    check(assert_type(ndarray_td64 == td, Any), np.ndarray)
    check(assert_type(ndarray_td64 != td, Any), np.ndarray)
    check(assert_type(np.timedelta64(1, "D") != td, Any), np.bool_)


# def test_interval() -> None:
#     i0 = pd.Interval(0, 1, closed="left")
#     i1 = pd.Interval(0.0, 1.0, closed="right")
#     i2 = pd.Interval(
#         pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02"), closed="both"
#     )
#     i3 = pd.Interval(pd.Timedelta("1 days"), pd.Timedelta("2 days"), closed="neither")
#     check(assert_type(i0, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(i1, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i2, "pd.Interval[pd.Timestamp]"), pd.Interval)
#     check(assert_type(i3, "pd.Interval[pd.Timedelta]"), pd.Interval)
#
#     check(assert_type(i0.closed, Literal["left", "right", "both", "neither"]), str)
#     check(assert_type(i0.closed_left, bool), bool)
#     check(assert_type(i0.closed_right, bool), bool)
#     check(assert_type(i0.is_empty, bool), bool)
#     check(assert_type(i0.left, int), int)
#     check(assert_type(i0.length, int), int)
#     check(assert_type(i0.mid, float), float)
#     check(assert_type(i0.open_left, bool), bool)
#     check(assert_type(i0.open_right, bool), bool)
#     check(assert_type(i0.right, int), int)
#
#     check(assert_type(i1.closed, Literal["left", "right", "both", "neither"]), str)
#     check(assert_type(i1.closed_left, bool), bool)
#     check(assert_type(i1.closed_right, bool), bool)
#     check(assert_type(i1.is_empty, bool), bool)
#     check(assert_type(i1.left, float), float)
#     check(assert_type(i1.length, float), float)
#     check(assert_type(i1.mid, float), float)
#     check(assert_type(i1.open_left, bool), bool)
#     check(assert_type(i1.open_right, bool), bool)
#     check(assert_type(i1.right, float), float)
#
#     check(assert_type(i2.closed, Literal["left", "right", "both", "neither"]), str)
#     check(assert_type(i2.closed_left, bool), bool)
#     check(assert_type(i2.closed_right, bool), bool)
#     check(assert_type(i2.is_empty, bool), bool)
#     check(assert_type(i2.left, pd.Timestamp), pd.Timestamp)
#     check(assert_type(i2.length, pd.Timedelta), pd.Timedelta)
#     check(assert_type(i2.mid, pd.Timestamp), pd.Timestamp)
#     check(assert_type(i2.open_left, bool), bool)
#     check(assert_type(i2.open_right, bool), bool)
#     check(assert_type(i2.right, pd.Timestamp), pd.Timestamp)
#
#     check(assert_type(i3.closed, Literal["left", "right", "both", "neither"]), str)
#     check(assert_type(i3.closed_left, bool), bool)
#     check(assert_type(i3.closed_right, bool), bool)
#     check(assert_type(i3.is_empty, bool), bool)
#     check(assert_type(i3.left, pd.Timedelta), pd.Timedelta)
#     check(assert_type(i3.length, pd.Timedelta), pd.Timedelta)
#     check(assert_type(i3.mid, pd.Timedelta), pd.Timedelta)
#     check(assert_type(i3.open_left, bool), bool)
#     check(assert_type(i3.open_right, bool), bool)
#     check(assert_type(i3.right, pd.Timedelta), pd.Timedelta)
#
#     check(assert_type(i0.overlaps(pd.Interval(0.5, 1.5, closed="left")), bool), bool)
#     check(assert_type(i0.overlaps(pd.Interval(2, 3, closed="left")), bool), bool)
#
#     check(assert_type(i1.overlaps(pd.Interval(0.5, 1.5, closed="left")), bool), bool)
#     check(assert_type(i1.overlaps(pd.Interval(2, 3, closed="left")), bool), bool)
#
#     check(
#         assert_type(
#             i2.overlaps(
#                 pd.Interval(
#                     pd.Timestamp(year=2017, month=1, day=1),
#                     pd.Timestamp(year=2017, month=1, day=2),
#                     closed="left",
#                 )
#             ),
#             bool,
#         ),
#         bool,
#     )
#     check(
#         assert_type(
#             i3.overlaps(
#                 pd.Interval(pd.Timedelta(days=1), pd.Timedelta(days=3), closed="left")
#             ),
#             bool,
#         ),
#         bool,
#     )
#
#     check(assert_type(i0 * 3, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(i1 * 3, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i3 * 3, "pd.Interval[pd.Timedelta]"), pd.Interval)
#
#     check(assert_type(i0 * 3.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i1 * 3.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i3 * 3.5, pd.Interval), pd.Interval)
#
#     check(assert_type(3 * i0, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(3 * i1, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(3 * i3, pd.Interval), pd.Interval)
#
#     check(assert_type(3.5 * i0, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(3.5 * i1, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(3.5 * i3, pd.Interval), pd.Interval)
#
#     check(assert_type(i0 / 3, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(i1 / 3, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i3 / 3, "pd.Interval[pd.Timedelta]"), pd.Interval)
#
#     check(assert_type(i0 / 3.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i1 / 3.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i3 / 3.5, "pd.Interval[pd.Timedelta]"), pd.Interval)
#
#     check(assert_type(i0 // 3, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(i1 // 3, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i3 // 3, "pd.Interval[pd.Timedelta]"), pd.Interval)
#
#     check(assert_type(i0 // 3.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i1 // 3.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i3 // 3.5, pd.Interval), pd.Interval)
#
#     check(assert_type(i0 - 1, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(i1 - 1, "pd.Interval[float]"), pd.Interval)
#     check(
#         assert_type(i2 - pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
#     )
#     check(
#         assert_type(i3 - pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
#     )
#
#     check(assert_type(i0 - 1.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i1 - 1.5, "pd.Interval[float]"), pd.Interval)
#     check(
#         assert_type(i2 - pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
#     )
#     check(
#         assert_type(i3 - pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
#     )
#
#     check(assert_type(i0 + 1, "pd.Interval[int]"), pd.Interval)
#     check(assert_type(i1 + 1, "pd.Interval[float]"), pd.Interval)
#     check(
#         assert_type(i2 + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
#     )
#     check(
#         assert_type(i3 + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
#     )
#
#     check(assert_type(i0 + 1.5, "pd.Interval[float]"), pd.Interval)
#     check(assert_type(i1 + 1.5, "pd.Interval[float]"), pd.Interval)
#     check(
#         assert_type(i2 + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
#     )
#     check(
#         assert_type(i3 + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
#     )
#
#     check(assert_type(i0 in i0, bool), bool)
#     check(assert_type(i1 in i0, bool), bool)
#     check(assert_type(i2 in i2, bool), bool)
#     check(assert_type(i3 in i3, bool), bool)
#
#     check(assert_type(hash(i0), int), int)
#     check(assert_type(hash(i1), int), int)
#     check(assert_type(hash(i2), int), int)
#     check(assert_type(hash(i3), int), int)
#
#
def test_timestamp() -> None:

    pd.Timestamp("2000-1-1")
    pd.Timestamp("2000-1-1", tz="US/Pacific")
    pd.Timestamp("2000-1-1", tz=pytz.timezone("US/Eastern"))
    pd.Timestamp("2000-1-1", tz=dateutil.tz.UTC)
    pd.Timestamp(
        year=2000,
        month=1,
        day=1,
        hour=1,
        minute=1,
        second=1,
        microsecond=1,
        nanosecond=1,
    )
    pd.Timestamp(1, unit="D")
    pd.Timestamp(1, unit="h")
    pd.Timestamp(1, unit="m")
    pd.Timestamp(1, unit="s")
    pd.Timestamp(1, unit="ms")
    pd.Timestamp(1, unit="us")
    pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27, fold=0)
    pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27, fold=1)
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
    )
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)
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

    ts = pd.Timestamp("2000-1-1")
    check(assert_type(ts + pd.Timedelta(days=1), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts + dt.timedelta(days=1), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.Timedelta(days=1) + ts, pd.Timestamp), pd.Timestamp)
    check(assert_type(dt.timedelta(days=1) + ts, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts + 3 * Day(), pd.Timestamp), pd.Timestamp)
    check(assert_type(3 * Day() + ts, pd.Timestamp), pd.Timestamp)
    check(
        assert_type(ts + pd.TimedeltaIndex([1, 2, 3], "D"), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(ts + pd.Series([1, 2], dtype="timedelta64[ns]"), "TimestampSeries"),
        pd.Series,
    )
    np_td64_arr: npt.NDArray[np.timedelta64] = np.array([1, 2], dtype="timedelta64[ns]")
    np_dt64_arr: npt.NDArray[np.datetime64] = np.array(
        [1, 2, 3], dtype="datetime64[ns]"
    )
    check(
        assert_type(ts + np_td64_arr, npt.NDArray[np.datetime64]),
        np.ndarray,
    )
    check(
        assert_type(pd.TimedeltaIndex([1, 2, 3], "D") + ts, pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(pd.Series([1, 2], dtype="timedelta64[ns]") + ts, "TimestampSeries"),
        pd.Series,
    )

    check(assert_type(ts - pd.Timedelta(days=1), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - dt.timedelta(days=1), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - 3 * Day(), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(ts - pd.TimedeltaIndex([1, 2, 3], "D"), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    ts_series = cast("TimedeltaSeries", pd.Series([1, 2], dtype="timedelta64[ns]"))
    check(
        assert_type(ts - ts_series, "TimestampSeries"),
        pd.Series,
    )
    check(assert_type(ts - np_td64_arr, npt.NDArray[np.datetime64]), np.ndarray)

    check(assert_type(ts > ts, bool), bool)
    check(assert_type(ts > np.datetime64(1, "ns"), bool), bool)
    check(assert_type(ts > dt.datetime(year=2000, month=1, day=1), bool), bool)
    check(assert_type(ts > pd.DatetimeIndex(["2000-1-1"]), np_ndarray_bool), np.ndarray)
    check(assert_type(ts > np_dt64_arr, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            ts > pd.Series([1, 2, 3], dtype="datetime64[ns]"), "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(dt.datetime(year=2000, month=1, day=1) > ts, bool), bool)
    check(assert_type(pd.DatetimeIndex(["2000-1-1"]) > ts, np_ndarray_bool), np.ndarray)
    check(assert_type(np_dt64_arr > ts, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            pd.Series([1, 2, 3], dtype="datetime64[ns]") > ts, "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(ts >= ts, bool), bool)
    check(assert_type(ts >= np.datetime64(1, "ns"), bool), bool)
    check(assert_type(ts >= dt.datetime(year=2000, month=1, day=1), bool), bool)
    check(
        assert_type(ts >= pd.DatetimeIndex(["2000-1-1"]), np_ndarray_bool), np.ndarray
    )
    check(assert_type(ts >= np_dt64_arr, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            ts >= pd.Series([1, 2, 3], dtype="datetime64[ns]"), "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(dt.datetime(year=2000, month=1, day=1) >= ts, bool), bool)
    check(
        assert_type(pd.DatetimeIndex(["2000-1-1"]) >= ts, np_ndarray_bool), np.ndarray
    )
    check(assert_type(np_dt64_arr >= ts, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            pd.Series([1, 2, 3], dtype="datetime64[ns]") >= ts, "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(ts < ts, bool), bool)
    check(assert_type(ts < np.datetime64(1, "ns"), bool), bool)
    check(assert_type(ts < dt.datetime(year=2000, month=1, day=1), bool), bool)
    check(assert_type(ts < pd.DatetimeIndex(["2000-1-1"]), np_ndarray_bool), np.ndarray)
    check(assert_type(ts < np_dt64_arr, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            ts < pd.Series([1, 2, 3], dtype="datetime64[ns]"), "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(dt.datetime(year=2000, month=1, day=1) < ts, bool), bool)
    check(assert_type(pd.DatetimeIndex(["2000-1-1"]) < ts, np_ndarray_bool), np.ndarray)
    check(assert_type(np_dt64_arr < ts, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            pd.Series([1, 2, 3], dtype="datetime64[ns]") < ts, "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(ts <= ts, bool), bool)
    check(assert_type(ts <= np.datetime64(1, "ns"), bool), bool)
    check(assert_type(ts <= dt.datetime(year=2000, month=1, day=1), bool), bool)
    check(
        assert_type(ts <= pd.DatetimeIndex(["2000-1-1"]), np_ndarray_bool), np.ndarray
    )
    check(assert_type(ts <= np_dt64_arr, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            ts <= pd.Series([1, 2, 3], dtype="datetime64[ns]"), "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(dt.datetime(year=2000, month=1, day=1) <= ts, bool), bool)
    check(
        assert_type(pd.DatetimeIndex(["2000-1-1"]) <= ts, np_ndarray_bool), np.ndarray
    )
    check(assert_type(np_dt64_arr <= ts, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            pd.Series([1, 2, 3], dtype="datetime64[ns]") <= ts, "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(ts == ts, bool), bool)
    check(assert_type(ts == np.datetime64(1, "ns"), bool), bool)
    check(assert_type(ts == dt.datetime(year=2000, month=1, day=1), bool), bool)
    check(
        assert_type(ts == pd.DatetimeIndex(["2000-1-1"]), np_ndarray_bool), np.ndarray
    )
    check(
        assert_type(ts == np_dt64_arr, np_ndarray_bool),
        np.ndarray,
    )
    check(
        assert_type(
            ts == pd.Series([1, 2, 3], dtype="datetime64[ns]"), "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(dt.datetime(year=2000, month=1, day=1) == ts, bool), bool)
    check(
        assert_type(pd.DatetimeIndex(["2000-1-1"]) == ts, np_ndarray_bool), np.ndarray
    )

    check(
        assert_type(
            pd.Series([1, 2, 3], dtype="datetime64[ns]") == ts, "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(ts != ts, bool), bool)
    check(assert_type(ts != np.datetime64(1, "ns"), bool), bool)
    check(assert_type(ts != dt.datetime(year=2000, month=1, day=1), bool), bool)
    check(
        assert_type(ts != pd.DatetimeIndex(["2000-1-1"]), np_ndarray_bool), np.ndarray
    )
    check(assert_type(ts != np_dt64_arr, np_ndarray_bool), np.ndarray)
    check(
        assert_type(
            ts != pd.Series([1, 2, 3], dtype="datetime64[ns]"), "pd.Series[bool]"
        ),
        pd.Series,
    )

    check(assert_type(dt.datetime(year=2000, month=1, day=1) != ts, bool), bool)
    check(
        assert_type(pd.DatetimeIndex(["2000-1-1"]) != ts, np_ndarray_bool), np.ndarray
    )
    check(
        assert_type(
            pd.Series([1, 2, 3], dtype="datetime64[ns]") != ts, "pd.Series[bool]"
        ),
        pd.Series,
    )

    # Failures due to NumPy ops returning Any
    check(
        assert_type(  # type: ignore[assert-type]
            np_td64_arr + ts, npt.NDArray[np.datetime64]  # type: ignore[operator]
        ),
        np.ndarray,
    )
    check(assert_type(np.datetime64(1, "ns") > ts, np.bool_), np.bool_)  # type: ignore[assert-type]
    check(assert_type(np.datetime64(1, "ns") >= ts, np.bool_), np.bool_)  # type: ignore[assert-type]
    check(assert_type(np.datetime64(1, "ns") < ts, np.bool_), np.bool_)  # type: ignore[assert-type]
    check(assert_type(np.datetime64(1, "ns") <= ts, np.bool_), np.bool_)  # type: ignore[assert-type]
    check(assert_type(np.datetime64(1, "ns") == ts, np.bool_), np.bool_)  # type: ignore[assert-type]
    check(assert_type(np_dt64_arr == ts, np_ndarray_bool), np.ndarray)  # type: ignore[assert-type]
    check(assert_type(np.datetime64(1, "ns") != ts, np.bool_), np.bool_)  # type: ignore[assert-type]
    check(assert_type(np_dt64_arr != ts, np_ndarray_bool), np.ndarray)  # type: ignore[assert-type]


def test_types_init() -> None:
    ts: pd.Timestamp = pd.Timestamp("2021-03-01T12")
    ts1: pd.Timestamp = pd.Timestamp(dt.date(2021, 3, 15))
    ts2: pd.Timestamp = pd.Timestamp(dt.datetime(2021, 3, 10, 12))
    ts3: pd.Timestamp = pd.Timestamp(pd.Timestamp("2021-03-01T12"))
    ts4: pd.Timestamp = pd.Timestamp(1515590000.1, unit="s")
    ts5: pd.Timestamp = pd.Timestamp(1515590000.1, unit="s", tz="US/Pacific")
    ts6: pd.Timestamp = pd.Timestamp(1515590000100000000)  # plain integer (nanosecond)
    ts7: pd.Timestamp = pd.Timestamp(2021, 3, 10, 12)
    ts8: pd.Timestamp = pd.Timestamp(year=2021, month=3, day=10, hour=12)
    ts9: pd.Timestamp = pd.Timestamp(
        year=2021, month=3, day=10, hour=12, tz="US/Pacific"
    )


def test_types_arithmetic() -> None:
    ts: pd.Timestamp = pd.to_datetime("2021-03-01")
    ts2: pd.Timestamp = pd.to_datetime("2021-01-01")
    delta: pd.Timedelta = pd.to_timedelta("1 day")

    check(assert_type(ts - ts2, pd.Timedelta), pd.Timedelta)
    check(assert_type(ts + delta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - delta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - dt.datetime(2021, 1, 3), pd.Timedelta), pd.Timedelta)


def test_types_comparison() -> None:
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
    check(assert_type(s, "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(ts2 <= s, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts2 >= s, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts2 < s, "pd.Series[bool]"), pd.Series, bool)
    check(assert_type(ts2 > s, "pd.Series[bool]"), pd.Series, bool)


def test_types_pydatetime() -> None:
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
