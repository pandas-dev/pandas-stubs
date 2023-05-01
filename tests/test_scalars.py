from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
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
from pandas._libs.tslibs.timedeltas import Components

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

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
    TimedeltaSeries: TypeAlias = pd.Series
    TimestampSeries: TypeAlias = pd.Series
    PeriodSeries: TypeAlias = pd.Series
    OffsetSeries: TypeAlias = pd.Series


def test_interval() -> None:
    interval_i = pd.Interval(0, 1, closed="left")
    interval_f = pd.Interval(0.0, 1.0, closed="right")
    interval_ts = pd.Interval(
        pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02"), closed="both"
    )
    interval_td = pd.Interval(
        pd.Timedelta("1 days"), pd.Timedelta("2 days"), closed="neither"
    )

    check(assert_type(interval_i, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(interval_f, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_ts, "pd.Interval[pd.Timestamp]"), pd.Interval, pd.Timestamp
    )
    check(
        assert_type(interval_td, "pd.Interval[pd.Timedelta]"), pd.Interval, pd.Timedelta
    )

    check(
        assert_type(interval_i.closed, Literal["left", "right", "both", "neither"]), str
    )
    check(assert_type(interval_i.closed_left, bool), bool)
    check(assert_type(interval_i.closed_right, bool), bool)
    check(assert_type(interval_i.is_empty, bool), bool)
    check(assert_type(interval_i.left, int), int)
    check(assert_type(interval_i.length, int), int)
    check(assert_type(interval_i.mid, float), float)
    check(assert_type(interval_i.open_left, bool), bool)
    check(assert_type(interval_i.open_right, bool), bool)
    check(assert_type(interval_i.right, int), int)

    check(
        assert_type(interval_f.closed, Literal["left", "right", "both", "neither"]), str
    )
    check(assert_type(interval_f.closed_left, bool), bool)
    check(assert_type(interval_f.closed_right, bool), bool)
    check(assert_type(interval_f.is_empty, bool), bool)
    check(assert_type(interval_f.left, float), float)
    check(assert_type(interval_f.length, float), float)
    check(assert_type(interval_f.mid, float), float)
    check(assert_type(interval_f.open_left, bool), bool)
    check(assert_type(interval_f.open_right, bool), bool)
    check(assert_type(interval_f.right, float), float)

    check(
        assert_type(interval_ts.closed, Literal["left", "right", "both", "neither"]),
        str,
    )
    check(assert_type(interval_ts.closed_left, bool), bool)
    check(assert_type(interval_ts.closed_right, bool), bool)
    check(assert_type(interval_ts.is_empty, bool), bool)
    check(assert_type(interval_ts.left, pd.Timestamp), pd.Timestamp)
    check(assert_type(interval_ts.length, pd.Timedelta), pd.Timedelta)
    check(assert_type(interval_ts.mid, pd.Timestamp), pd.Timestamp)
    check(assert_type(interval_ts.open_left, bool), bool)
    check(assert_type(interval_ts.open_right, bool), bool)
    check(assert_type(interval_ts.right, pd.Timestamp), pd.Timestamp)

    check(
        assert_type(interval_td.closed, Literal["left", "right", "both", "neither"]),
        str,
    )
    check(assert_type(interval_td.closed_left, bool), bool)
    check(assert_type(interval_td.closed_right, bool), bool)
    check(assert_type(interval_td.is_empty, bool), bool)
    check(assert_type(interval_td.left, pd.Timedelta), pd.Timedelta)
    check(assert_type(interval_td.length, pd.Timedelta), pd.Timedelta)
    check(assert_type(interval_td.mid, pd.Timedelta), pd.Timedelta)
    check(assert_type(interval_td.open_left, bool), bool)
    check(assert_type(interval_td.open_right, bool), bool)
    check(assert_type(interval_td.right, pd.Timedelta), pd.Timedelta)

    check(
        assert_type(interval_i.overlaps(pd.Interval(0.5, 1.5, closed="left")), bool),
        bool,
    )
    check(
        assert_type(interval_i.overlaps(pd.Interval(2, 3, closed="left")), bool), bool
    )

    check(
        assert_type(interval_f.overlaps(pd.Interval(0.5, 1.5, closed="left")), bool),
        bool,
    )
    check(
        assert_type(interval_f.overlaps(pd.Interval(2, 3, closed="left")), bool), bool
    )
    ts1 = pd.Timestamp(year=2017, month=1, day=1)
    ts2 = pd.Timestamp(year=2017, month=1, day=2)
    check(
        assert_type(interval_ts.overlaps(pd.Interval(ts1, ts2, closed="left")), bool),
        bool,
    )
    td1 = pd.Timedelta(days=1)
    td2 = pd.Timedelta(days=3)
    check(
        assert_type(interval_td.overlaps(pd.Interval(td1, td2, closed="left")), bool),
        bool,
    )


def test_interval_math() -> None:
    interval_i = pd.Interval(0, 1, closed="left")
    interval_f = pd.Interval(0.0, 1.0, closed="right")
    interval_ts = pd.Interval(
        pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02"), closed="both"
    )
    interval_td = pd.Interval(
        pd.Timedelta("1 days"), pd.Timedelta("2 days"), closed="neither"
    )

    check(assert_type(interval_i * 3, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(interval_f * 3, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_td * 3, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(interval_i * 3.5, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(interval_f * 3.5, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_td * 3.5, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(3 * interval_i, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(3 * interval_f, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(3 * interval_td, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(3.5 * interval_i, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(3.5 * interval_f, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(3.5 * interval_td, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(interval_i / 3, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(interval_f / 3, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_td / 3, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(interval_i / 3.5, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(interval_f / 3.5, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_td / 3.5, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(interval_i // 3, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(interval_f // 3, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_td // 3, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(interval_i // 3.5, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(interval_f // 3.5, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_td // 3.5, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    # Subtraction
    check(assert_type(interval_i - 1, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(interval_f - 1, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_ts - pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"),
        pd.Interval,
        pd.Timestamp,
    )
    check(
        assert_type(interval_td - pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"),
        pd.Interval,
        pd.Timedelta,
    )

    check(assert_type(interval_i - 1.5, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(interval_f - 1.5, "pd.Interval[float]"), pd.Interval, float)
    check(
        assert_type(interval_ts - pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(interval_td - pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )

    # Addition
    check(assert_type(interval_i + 1, "pd.Interval[int]"), pd.Interval)
    check(assert_type(1 + interval_i, "pd.Interval[int]"), pd.Interval)
    check(assert_type(interval_f + 1, "pd.Interval[float]"), pd.Interval)
    check(assert_type(1 + interval_f, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(interval_ts + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(pd.Timedelta(days=1) + interval_ts, "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(interval_td + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )
    check(
        assert_type(pd.Timedelta(days=1) + interval_td, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )

    check(assert_type(interval_i + 1.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(1.5 + interval_i, "pd.Interval[float]"), pd.Interval)
    check(assert_type(interval_f + 1.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(1.5 + interval_f, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(interval_ts + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(pd.Timedelta(days=1) + interval_ts, "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(interval_td + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )
    check(
        assert_type(pd.Timedelta(days=1) + interval_td, "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )


def test_interval_cmp():
    interval_i = pd.Interval(0, 1, closed="left")
    interval_f = pd.Interval(0.0, 1.0, closed="right")
    interval_ts = pd.Interval(
        pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02"), closed="both"
    )
    interval_td = pd.Interval(
        pd.Timedelta("1 days"), pd.Timedelta("2 days"), closed="neither"
    )

    check(assert_type(0.5 in interval_i, bool), bool)
    check(assert_type(1 in interval_i, bool), bool)
    check(assert_type(1 in interval_f, bool), bool)
    check(assert_type(pd.Timestamp("2000-1-1") in interval_ts, bool), bool)
    check(assert_type(pd.Timedelta(days=1) in interval_td, bool), bool)

    interval_index_int = pd.IntervalIndex([interval_i])
    check(
        assert_type(interval_index_int >= interval_i, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_index_int < interval_i, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_index_int <= interval_i, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_index_int > interval_i, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )

    check(
        assert_type(interval_i >= interval_index_int, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_i < interval_index_int, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_i <= interval_index_int, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_i > interval_index_int, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )

    check(
        assert_type(interval_index_int == interval_i, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(interval_index_int != interval_i, np_ndarray_bool),
        np.ndarray,
        np.bool_,
    )

    check(
        assert_type(
            interval_i == interval_index_int,
            np_ndarray_bool,
        ),
        np.ndarray,
        np.bool_,
    )
    check(
        assert_type(
            interval_i != interval_index_int,
            np_ndarray_bool,
        ),
        np.ndarray,
        np.bool_,
    )


def test_timedelta_construction() -> None:
    check(assert_type(pd.Timedelta(1, "H"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "T"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "S"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "L"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "U"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "N"), pd.Timedelta), pd.Timedelta)
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
    check(
        assert_type(
            pd.Timedelta(
                days=1,
                seconds=1,
                microseconds=1,
                minutes=1,
                hours=1,
                weeks=1,
                milliseconds=1,
            ),
            pd.Timedelta,
        ),
        pd.Timedelta,
    )


def test_timedelta_properties_methods() -> None:
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


def test_timedelta_add_sub() -> None:
    td = pd.Timedelta("1 day")

    ndarray_td64: npt.NDArray[np.timedelta64] = np.array(
        [1, 2, 3], dtype="timedelta64[D]"
    )
    ndarray_dt64: npt.NDArray[np.datetime64] = np.array(
        [1, 2, 3], dtype="datetime64[D]"
    )
    as_period = pd.Period("2012-01-01", freq="D")
    as_timestamp = pd.Timestamp("2012-01-01")
    as_datetime = dt.datetime(2012, 1, 1)
    as_date = dt.date(2012, 1, 1)
    as_datetime64 = np.datetime64(1, "ns")
    as_dt_timedelta = dt.timedelta(days=1)
    as_timedelta64 = np.timedelta64(1, "D")
    as_timedelta_index = pd.TimedeltaIndex([td])
    as_timedelta_series = pd.Series(as_timedelta_index)
    as_period_index = pd.period_range("2012-01-01", periods=3, freq="D")
    as_datetime_index = pd.date_range("2012-01-01", periods=3)
    as_ndarray_td64 = ndarray_td64
    as_ndarray_dt64 = ndarray_dt64
    as_nat = pd.NaT

    check(assert_type(td + td, pd.Timedelta), pd.Timedelta)
    check(assert_type(td + as_period, pd.Period), pd.Period)
    check(assert_type(td + as_timestamp, pd.Timestamp), pd.Timestamp)
    check(assert_type(td + as_datetime, pd.Timestamp), pd.Timestamp)
    check(assert_type(td + as_date, dt.date), dt.date)
    check(assert_type(td + as_datetime64, pd.Timestamp), pd.Timestamp)
    check(assert_type(td + as_dt_timedelta, pd.Timedelta), pd.Timedelta)
    check(assert_type(td + as_timedelta64, pd.Timedelta), pd.Timedelta)
    check(assert_type(td + as_timedelta_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(
        assert_type(td + as_timedelta_series, TimedeltaSeries), pd.Series, pd.Timedelta
    )
    check(assert_type(td + as_period_index, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(td + as_datetime_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(td + as_ndarray_td64, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(
        assert_type(td + as_ndarray_dt64, npt.NDArray[np.datetime64]),
        np.ndarray,
        np.datetime64,
    )
    check(assert_type(td + as_nat, NaTType), NaTType)

    check(assert_type(as_period + td, pd.Period), pd.Period)
    check(assert_type(as_timestamp + td, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_datetime + td, dt.datetime), dt.datetime)
    check(assert_type(as_date + td, dt.date), dt.date)
    check(assert_type(as_datetime64 + td, pd.Timestamp), pd.Timestamp)
    # pyright can't know that as_td_timedelta + td calls
    # td.__radd__(as_td_timedelta),  not timedelta.__add__
    # https://github.com/microsoft/pyright/issues/4088
    check(
        assert_type(
            as_dt_timedelta + td,  # pyright: ignore[reportGeneralTypeIssues]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(assert_type(as_timedelta64 + td, pd.Timedelta), pd.Timedelta)
    check(assert_type(as_timedelta_index + td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(
        assert_type(as_timedelta_series + td, TimedeltaSeries), pd.Series, pd.Timedelta
    )
    check(assert_type(as_period_index + td, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(as_datetime_index + td, pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(
            as_ndarray_td64 + td,
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
        np.timedelta64,
    )
    check(assert_type(as_nat + td, NaTType), NaTType)

    # sub is not symmetric with dates. In general date_like - timedelta is
    # sensible, while timedelta - date_like is not
    # TypeError: as_period, as_timestamp, as_datetime, as_date, as_datetime64,
    #            as_period_index, as_datetime_index, as_ndarray_dt64
    if TYPE_CHECKING_INVALID_USAGE:
        td - as_period  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_timestamp  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_datetime  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_date  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_datetime64  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_period_index  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_datetime_index  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        td - as_ndarray_dt64  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]

    check(assert_type(td - td, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as_dt_timedelta, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as_timedelta64, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as_timedelta_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(
        assert_type(td - as_timedelta_series, TimedeltaSeries), pd.Series, pd.Timedelta
    )
    check(
        assert_type(td - as_ndarray_td64, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(assert_type(td - as_nat, NaTType), NaTType)
    check(assert_type(as_period - td, pd.Period), pd.Period)
    check(assert_type(as_timestamp - td, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_datetime - td, dt.datetime), dt.datetime)
    check(assert_type(as_date - td, dt.date), dt.date)
    check(assert_type(as_datetime64 - td, pd.Timestamp), pd.Timestamp)
    # pyright can't know that as_dt_timedelta - td calls td.__rsub__(as_dt_timedelta),
    # not as_dt_timedelta.__sub__
    # https://github.com/microsoft/pyright/issues/4088
    check(
        assert_type(
            as_dt_timedelta - td,  # pyright: ignore[reportGeneralTypeIssues]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(assert_type(as_timedelta64 - td, pd.Timedelta), pd.Timedelta)
    check(assert_type(as_timedelta_index - td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(
        assert_type(as_timedelta_series - td, TimedeltaSeries), pd.Series, pd.Timedelta
    )
    check(assert_type(as_period_index - td, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(as_datetime_index - td, pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(
            as_ndarray_td64 - td,
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
        np.timedelta64,
    )
    check(
        assert_type(
            as_ndarray_dt64 - td,
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
        np.datetime64,
    )
    check(assert_type(as_nat - td, NaTType), NaTType)


def test_timedelta_mul_div() -> None:
    td = pd.Timedelta("1 day")

    i_idx = pd.Index([1, 2, 3], dtype=int)
    f_idx = pd.Index([1.2, 2.2, 3.4], dtype=float)

    np_intp_arr: npt.NDArray[np.integer] = np.array([1, 2, 3])
    np_float_arr: npt.NDArray[np.floating] = np.array([1.2, 2.2, 3.4])

    md_int = 3
    md_float = 3.5
    md_ndarray_intp = np_intp_arr
    md_ndarray_float = np_float_arr
    mp_series_int = pd.Series([1, 2, 3], dtype=int)
    md_series_float = pd.Series([1.2, 2.2, 3.4], dtype=float)
    md_int64_index = i_idx
    md_float_index = f_idx
    md_timedelta_series = pd.Series(pd.timedelta_range("1 day", periods=3))

    check(assert_type(td * md_int, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * md_float, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td * md_ndarray_intp, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(
        assert_type(td * md_ndarray_float, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(assert_type(td * mp_series_int, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td * md_series_float, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td * md_int64_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td * md_float_index, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(md_int * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(md_float * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(md_ndarray_intp * td, np.ndarray), np.ndarray, np.timedelta64)
    check(assert_type(md_ndarray_float * td, np.ndarray), np.ndarray, np.timedelta64)
    check(assert_type(mp_series_int * td, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(md_series_float * td, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(md_int64_index * td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(md_float_index * td, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(td // td, int), int)
    check(assert_type(td // pd.NaT, float), float)
    check(assert_type(td // md_int, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // md_float, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td // md_ndarray_intp, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(
        assert_type(td // md_ndarray_float, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(assert_type(td // mp_series_int, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td // md_series_float, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td // md_int64_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td // md_float_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(
        assert_type(td // md_timedelta_series, "pd.Series[int]"),
        pd.Series,
        np.longlong,
    )

    check(assert_type(pd.NaT // td, float), float)
    # Note: None of the reverse floordiv work
    # TypeError: md_int, md_float, md_ndarray_intp, md_ndarray_float, mp_series_int,
    #            mp_series_float, md_int64_index, md_float_index
    if TYPE_CHECKING_INVALID_USAGE:
        md_int // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_float // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_ndarray_intp // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_ndarray_float // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        mp_series_int // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_series_float // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_int64_index // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_float_index // td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]

    check(assert_type(td / td, float), float)
    check(assert_type(td / pd.NaT, float), float)
    check(assert_type(td / md_int, pd.Timedelta), pd.Timedelta)
    check(assert_type(td / md_float, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td / md_ndarray_intp, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(
        assert_type(td / md_ndarray_float, npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(assert_type(td / mp_series_int, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td / md_series_float, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td / md_int64_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td / md_float_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td / md_timedelta_series, "pd.Series[float]"), pd.Series, float)

    check(assert_type(pd.NaT / td, float), float)
    # Note: None of the reverse truediv work
    # TypeError: md_int, md_float, md_ndarray_intp, md_ndarray_float, mp_series_int,
    #            mp_series_float, md_int64_index, md_float_index
    if TYPE_CHECKING_INVALID_USAGE:
        md_int / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_float / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_ndarray_intp / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_ndarray_float / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        mp_series_int / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_series_float / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_int64_index / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        md_float_index / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]


def test_timedelta_mod_abs_unary() -> None:
    td = pd.Timedelta("1 day")

    i_idx = pd.Index([1, 2, 3], dtype=int)
    f_idx = pd.Index([1.2, 2.2, 3.4], dtype=float)

    check(assert_type(td % 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td % 3.5, pd.Timedelta), pd.Timedelta)
    check(assert_type(td % td, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td % np.array([1, 2, 3]), npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    check(
        assert_type(td % np.array([1.2, 2.2, 3.4]), npt.NDArray[np.timedelta64]),
        np.ndarray,
        np.timedelta64,
    )
    int_series = pd.Series([1, 2, 3], dtype=int)
    float_series = pd.Series([1.2, 2.2, 3.4], dtype=float)
    check(assert_type(td % int_series, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td % float_series, TimedeltaSeries), pd.Series, pd.Timedelta)
    check(assert_type(td % i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(
        assert_type(td % f_idx, pd.TimedeltaIndex),
        pd.TimedeltaIndex,
    )

    check(assert_type(abs(td), pd.Timedelta), pd.Timedelta)

    check(assert_type(td.__abs__(), pd.Timedelta), pd.Timedelta)
    check(assert_type(-td, pd.Timedelta), pd.Timedelta)
    check(assert_type(+td, pd.Timedelta), pd.Timedelta)


def test_timedelta_cmp() -> None:
    td = pd.Timedelta("1 day")
    ndarray_td64: npt.NDArray[np.timedelta64] = np.array(
        [1, 2, 3], dtype="timedelta64[D]"
    )
    c_timedelta = td
    c_dt_timedelta = dt.timedelta(days=1)
    c_timedelta64 = np.timedelta64(1, "D")
    c_ndarray_td64 = ndarray_td64
    c_timedelta_index = pd.TimedeltaIndex([1, 2, 3], unit="D")
    c_timedelta_series = pd.Series(pd.TimedeltaIndex([1, 2, 3]))

    check(assert_type(td < c_timedelta, bool), bool)
    check(assert_type(td < c_dt_timedelta, bool), bool)
    check(assert_type(td < c_timedelta64, bool), bool)
    check(assert_type(td < c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_timedelta_index < td, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_dt_timedelta < td, bool), bool)
    check(assert_type(c_ndarray_td64 < td, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_timedelta_index < td, np_ndarray_bool), np.ndarray, np.bool_)

    gt = check(assert_type(td > c_timedelta, bool), bool)
    le = check(assert_type(td <= c_timedelta, bool), bool)
    assert gt != le

    gt = check(assert_type(td > c_dt_timedelta, bool), bool)
    le = check(assert_type(td <= c_dt_timedelta, bool), bool)
    assert gt != le

    gt = check(assert_type(td > c_timedelta64, bool), bool)
    le = check(assert_type(td <= c_timedelta64, bool), bool)
    assert gt != le

    gt_a = check(
        assert_type(td > c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_
    )
    le_a = check(
        assert_type(td <= c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_a = check(
        assert_type(td > c_timedelta_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    le_a = check(
        assert_type(td <= c_timedelta_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_s = check(
        assert_type(td > c_timedelta_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    le_s = check(
        assert_type(td <= c_timedelta_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (gt_s != le_s).all()

    gt = check(assert_type(c_dt_timedelta > td, bool), bool)
    le = check(assert_type(c_dt_timedelta <= td, bool), bool)
    assert gt != le

    gt_b = check(assert_type(c_timedelta64 > td, Any), bool)
    le_b = check(assert_type(c_timedelta64 <= td, Any), bool)
    assert gt_b != le_b

    gt_a = check(
        assert_type(c_ndarray_td64 > td, np_ndarray_bool), np.ndarray, np.bool_
    )
    le_a = check(
        assert_type(c_ndarray_td64 <= td, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_a = check(
        assert_type(c_timedelta_index > td, np_ndarray_bool), np.ndarray, np.bool_
    )
    le_a = check(
        assert_type(c_timedelta_index <= td, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    eq_s = check(
        assert_type(c_timedelta_series > td, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(c_timedelta_series <= td, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()

    lt = check(assert_type(td < c_timedelta, bool), bool)
    ge = check(assert_type(td >= c_timedelta, bool), bool)
    assert lt != ge

    lt = check(assert_type(td < c_dt_timedelta, bool), bool)
    ge = check(assert_type(td >= c_dt_timedelta, bool), bool)
    assert lt != ge

    lt = check(assert_type(td < c_timedelta64, bool), bool)
    ge = check(assert_type(td >= c_timedelta64, bool), bool)
    assert lt != ge

    lt_a = check(
        assert_type(td < c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_
    )
    ge_a = check(
        assert_type(td >= c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    lt_a = check(
        assert_type(td < c_timedelta_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    ge_a = check(
        assert_type(td >= c_timedelta_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    eq_s = check(
        assert_type(td < c_timedelta_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(td >= c_timedelta_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()

    lt = check(assert_type(c_dt_timedelta < td, bool), bool)
    ge = check(assert_type(c_dt_timedelta >= td, bool), bool)
    assert lt != ge

    lt_b = check(assert_type(c_timedelta64 < td, Any), bool)
    ge_b = check(assert_type(c_timedelta64 >= td, Any), bool)
    assert lt_b != ge_b

    lt_a = check(
        assert_type(c_ndarray_td64 < td, np_ndarray_bool), np.ndarray, np.bool_
    )
    ge_a = check(
        assert_type(c_ndarray_td64 >= td, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    lt_a = check(
        assert_type(c_timedelta_index < td, np_ndarray_bool), np.ndarray, np.bool_
    )
    ge_a = check(
        assert_type(c_timedelta_index >= td, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    eq_s = check(
        assert_type(c_timedelta_series < td, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(c_timedelta_series >= td, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()

    eq = check(assert_type(td == td, bool), bool)
    ne = check(assert_type(td != td, bool), bool)
    assert eq != ne

    eq = check(assert_type(td == c_dt_timedelta, bool), bool)
    ne = check(assert_type(td != c_dt_timedelta, bool), bool)
    assert eq != ne

    eq = check(assert_type(td == c_timedelta64, bool), bool)
    ne = check(assert_type(td != c_timedelta64, bool), bool)
    assert eq != ne

    eq_a = check(
        assert_type(td == c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_a = check(
        assert_type(td != c_ndarray_td64, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_a != ne_a).all()

    eq_a = check(
        assert_type(td == c_timedelta_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_a = check(
        assert_type(td != c_timedelta_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_a != ne_a).all()

    eq_s = check(
        assert_type(td == c_timedelta_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(td != c_timedelta_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()

    eq = check(assert_type(td == 1, Literal[False]), bool)
    ne = check(assert_type(td != 1, Literal[True]), bool)
    assert eq != ne

    eq = check(assert_type(td == (3 + 2j), Literal[False]), bool)
    ne = check(assert_type(td != (3 + 2j), Literal[True]), bool)
    assert eq != ne


def test_timedelta_cmp_rhs() -> None:
    # Test that check eq and ne when Timedelta is the RHS argument
    # that use the __eq__ and __ne__ methods of the LHS
    td = pd.Timedelta("1 day")
    ndarray_td64: npt.NDArray[np.timedelta64] = np.array(
        [1, 2, 3], dtype="timedelta64[D]"
    )
    c_dt_timedelta = dt.timedelta(days=1)
    c_timedelta64 = np.timedelta64(1, "D")
    c_ndarray_td64 = ndarray_td64
    c_timedelta_index = pd.TimedeltaIndex([1, 2, 3], unit="D")
    c_timedelta_series = pd.Series(pd.TimedeltaIndex([1, 2, 3]))

    eq = check(assert_type(c_dt_timedelta == td, bool), bool)
    ne = check(assert_type(c_dt_timedelta != td, bool), bool)
    assert eq != ne

    eq = check(assert_type(c_timedelta64 == td, Any), bool)
    ne = check(assert_type(c_timedelta64 != td, Any), bool)
    assert eq != ne

    eq_a = check(assert_type(c_ndarray_td64 == td, Any), np.ndarray, np.bool_)
    ne_a = check(assert_type(c_ndarray_td64 != td, Any), np.ndarray, np.bool_)
    assert (eq_a != ne_a).all()

    eq_a = check(
        assert_type(c_timedelta_index == td, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_a = check(
        assert_type(c_timedelta_index != td, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_a != ne_a).all()

    eq_s = check(
        assert_type(c_timedelta_series == td, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(c_timedelta_series != td, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()


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
    check(assert_type(as_timedelta_series, TimedeltaSeries), pd.Series, pd.Timedelta)
    as_np_ndarray_td64 = np_td64_arr

    check(assert_type(ts + as_pd_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_pd_timedelta + ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts + as_dt_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_dt_timedelta + ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts + as_offset, pd.Timestamp), pd.Timestamp)
    check(assert_type(as_offset + ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts + as_timedelta_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(as_timedelta_index + ts, pd.DatetimeIndex), pd.DatetimeIndex)

    check(
        assert_type(ts + as_timedelta_series, TimestampSeries), pd.Series, pd.Timestamp
    )
    check(
        assert_type(as_timedelta_series + ts, TimestampSeries), pd.Series, pd.Timestamp
    )

    check(
        assert_type(ts + as_np_ndarray_td64, npt.NDArray[np.datetime64]),
        np.ndarray,
        np.datetime64,
    )
    check(
        assert_type(
            as_np_ndarray_td64 + ts,
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
        np.datetime64,
    )

    # Reverse order is not possible for all of these
    check(assert_type(ts - as_pd_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_dt_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_offset, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_timedelta_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(ts - as_timedelta_series, TimestampSeries), pd.Series, pd.Timestamp
    )
    check(
        assert_type(ts - as_np_ndarray_td64, npt.NDArray[np.datetime64]),
        np.ndarray,
        np.datetime64,
    )


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
    c_series_dt64: TimestampSeries = pd.Series([1, 2, 3], dtype="datetime64[ns]")
    c_series_timestamp = pd.Series(pd.DatetimeIndex(["2000-1-1"]))
    check(assert_type(c_series_timestamp, TimestampSeries), pd.Series, pd.Timestamp)
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

    check(assert_type(ts > c_datetimeindex, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(ts <= c_datetimeindex, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(ts > c_np_ndarray_dt64, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(ts <= c_np_ndarray_dt64, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(ts > c_series_timestamp, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts <= c_series_timestamp, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(ts > c_series_dt64, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts <= c_series_dt64, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(c_np_dt64 > ts, Any), bool)
    check(assert_type(c_np_dt64 <= ts, Any), bool)

    gt = check(assert_type(c_dt_datetime > ts, bool), bool)
    lte = check(assert_type(c_dt_datetime <= ts, bool), bool)
    assert gt != lte

    check(assert_type(c_datetimeindex > ts, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_datetimeindex <= ts, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(c_np_ndarray_dt64 > ts, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_np_ndarray_dt64 <= ts, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(c_series_dt64 > ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(c_series_dt64 <= ts, "pd.Series[bool]"), pd.Series, np.bool_)

    gte = check(assert_type(ts >= c_timestamp, bool), bool)
    lt = check(assert_type(ts < c_timestamp, bool), bool)
    assert gte != lt

    gte = check(assert_type(ts >= c_np_dt64, bool), bool)
    lt = check(assert_type(ts < c_np_dt64, bool), bool)
    assert gte != lt

    gte = check(assert_type(ts >= c_dt_datetime, bool), bool)
    lt = check(assert_type(ts < c_dt_datetime, bool), bool)
    assert gte != lt

    check(assert_type(ts >= c_datetimeindex, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(ts < c_datetimeindex, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(ts >= c_np_ndarray_dt64, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(ts < c_np_ndarray_dt64, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(ts >= c_series_timestamp, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts < c_series_timestamp, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(ts >= c_series_dt64, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts < c_series_dt64, "pd.Series[bool]"), pd.Series, np.bool_)

    gte = check(assert_type(c_dt_datetime >= ts, bool), bool)
    lt = check(assert_type(c_dt_datetime < ts, bool), bool)
    assert gte != lt

    check(assert_type(c_np_dt64 >= ts, Any), bool)
    check(assert_type(c_np_dt64 < ts, Any), bool)

    check(assert_type(c_datetimeindex >= ts, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_datetimeindex < ts, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(c_np_ndarray_dt64 >= ts, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c_np_ndarray_dt64 < ts, np_ndarray_bool), np.ndarray, np.bool_)

    check(assert_type(c_series_dt64 >= ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(c_series_dt64 < ts, "pd.Series[bool]"), pd.Series, np.bool_)

    eq = check(assert_type(ts == c_timestamp, bool), bool)
    ne = check(assert_type(ts != c_timestamp, bool), bool)
    assert eq != ne

    eq = check(assert_type(ts == c_np_dt64, bool), bool)
    ne = check(assert_type(ts != c_np_dt64, bool), bool)
    assert eq != ne

    eq = check(assert_type(ts == c_dt_datetime, bool), bool)
    ne = check(assert_type(ts != c_dt_datetime, bool), bool)
    assert eq != ne

    eq_arr = check(
        assert_type(ts == c_datetimeindex, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_arr = check(
        assert_type(ts != c_datetimeindex, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_arr != ne_arr).all()

    eq_arr = check(
        assert_type(ts == c_np_ndarray_dt64, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_arr = check(
        assert_type(ts != c_np_ndarray_dt64, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_arr != ne_arr).all()

    eq_s = check(
        assert_type(ts == c_series_timestamp, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(ts != c_series_timestamp, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()

    eq_s = check(
        assert_type(ts == c_series_dt64, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(ts != c_series_dt64, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()


def test_timestamp_eq_ne_rhs() -> None:
    # These test equality using the LHS objects __eq__ and __ne__ methods
    # The tests are retained for completeness, but are not strictly necessary
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)

    np_dt64_arr: npt.NDArray[np.datetime64] = np.array(
        [1, 2, 3], dtype="datetime64[ns]"
    )

    c_np_dt64 = np.datetime64(1, "ns")
    c_dt_datetime = dt.datetime(year=2000, month=1, day=1)
    c_datetimeindex = pd.DatetimeIndex(["2000-1-1"])
    c_np_ndarray_dt64 = np_dt64_arr
    c_series_dt64: pd.Series[pd.Timestamp] = pd.Series(
        [1, 2, 3], dtype="datetime64[ns]"
    )

    eq_a = check(assert_type(c_np_dt64 == ts, Any), bool)
    ne_a = check(assert_type(c_np_dt64 != ts, Any), bool)
    assert eq_a != ne_a

    eq = check(assert_type(c_dt_datetime == ts, bool), bool)
    ne = check(assert_type(c_dt_datetime != ts, bool), bool)
    assert eq != ne

    eq_arr = check(
        assert_type(c_datetimeindex == ts, np_ndarray_bool), np.ndarray, np.bool_
    )
    ne_arr = check(
        assert_type(c_datetimeindex != ts, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (eq_arr != ne_arr).all()

    eq_a = check(assert_type(c_np_ndarray_dt64 != ts, Any), np.ndarray, np.bool_)
    ne_a = check(assert_type(c_np_ndarray_dt64 == ts, Any), np.ndarray, np.bool_)
    assert (eq_a != ne_a).all()

    eq_s = check(
        assert_type(c_series_dt64 == ts, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(c_series_dt64 != ts, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()


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


def test_timestamp_misc_methods() -> None:
    ts = pd.Timestamp("2021-03-01T12")
    check(assert_type(ts, pd.Timestamp), pd.Timestamp)

    check(assert_type(ts.to_numpy(), np.datetime64), np.datetime64)

    check(assert_type(pd.Timestamp.fromtimestamp(432.54), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.Timestamp.fromtimestamp(432.54, tz="US/Pacific"), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(pd.Timestamp.fromordinal(700000), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.Timestamp.fromordinal(700000, tz="US/Pacific"), pd.Timestamp),
        pd.Timestamp,
    )

    ts2 = ts.replace(
        year=2020,
        month=2,
        day=2,
        hour=12,
        minute=21,
        second=21,
        microsecond=12,
        tzinfo=dateutil.tz.UTC,
        fold=0,
    )
    check(assert_type(ts2, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts.tz_localize("US/Pacific", False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts.tz_localize("US/Pacific", True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts.tz_localize("US/Pacific", "NaT"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(ts.tz_localize("US/Pacific", "raise"), pd.Timestamp), pd.Timestamp
    )

    check(
        assert_type(
            ts.tz_localize("US/Pacific", nonexistent="shift_forward"), pd.Timestamp
        ),
        pd.Timestamp,
    )
    check(
        assert_type(
            ts.tz_localize("US/Pacific", nonexistent="shift_backward"), pd.Timestamp
        ),
        pd.Timestamp,
    )
    check(
        assert_type(ts.tz_localize("US/Pacific", nonexistent="NaT"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts.tz_localize("US/Pacific", nonexistent="raise"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(
            ts.tz_localize("US/Pacific", nonexistent=pd.Timedelta("1D")), pd.Timestamp
        ),
        pd.Timestamp,
    )

    check(assert_type(ts2.round("1S"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1S", ambiguous="raise"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1S", ambiguous=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1S", ambiguous=False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1S", ambiguous="NaT"), pd.Timestamp), pd.Timestamp)

    check(
        assert_type(ts2.round("2H", nonexistent="shift_forward"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts2.round("2H", nonexistent="shift_backward"), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(ts2.round("2H", nonexistent="NaT"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("2H", nonexistent="raise"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(ts2.round("2H", nonexistent=pd.Timedelta(24, "H")), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts2.round("2H", nonexistent=dt.timedelta(hours=24)), pd.Timestamp),
        pd.Timestamp,
    )

    check(assert_type(ts2.ceil("1S"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1S", ambiguous="raise"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1S", ambiguous=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1S", ambiguous=False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1S", ambiguous="NaT"), pd.Timestamp), pd.Timestamp)

    check(
        assert_type(ts2.ceil("2H", nonexistent="shift_forward"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts2.ceil("2H", nonexistent="shift_backward"), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(ts2.ceil("2H", nonexistent="NaT"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("2H", nonexistent="raise"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(ts2.ceil("2H", nonexistent=pd.Timedelta(24, "H")), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts2.ceil("2H", nonexistent=dt.timedelta(hours=24)), pd.Timestamp),
        pd.Timestamp,
    )

    check(assert_type(ts2.floor("1S"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1S", ambiguous="raise"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1S", ambiguous=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1S", ambiguous=False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1S", ambiguous="NaT"), pd.Timestamp), pd.Timestamp)

    check(
        assert_type(ts2.floor("2H", nonexistent="shift_forward"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts2.floor("2H", nonexistent="shift_backward"), pd.Timestamp),
        pd.Timestamp,
    )
    check(assert_type(ts2.floor("2H", nonexistent="NaT"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("2H", nonexistent="raise"), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(ts2.floor("2H", nonexistent=pd.Timedelta(24, "H")), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(ts2.floor("2H", nonexistent=dt.timedelta(hours=24)), pd.Timestamp),
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
    check(assert_type(tssr, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(tssr2, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(tssr3, "pd.Series[bool]"), pd.Series, np.bool_)
    # GH 265
    data = pd.date_range("2022-01-01", "2022-01-31", freq="D")
    s = pd.Series(data)
    ts2 = pd.Timestamp("2022-01-15")
    check(assert_type(s, TimestampSeries), pd.Series, pd.Timestamp)
    check(assert_type(ts2 <= s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts2 >= s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts2 < s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts2 > s, "pd.Series[bool]"), pd.Series, np.bool_)


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


def test_timestamp_combine() -> None:
    ts = pd.Timestamp("2022-03-18")
    # mypy and pyright disagree from actual type due to inheritance.
    # Same issue with some timedelta ops
    check(
        assert_type(
            ts.combine(dt.date(2000, 1, 1), dt.time(12, 21, 21, 12)), dt.datetime
        ),
        pd.Timestamp,
    )


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
    check(assert_type(as_td_series, TimedeltaSeries), pd.Series, pd.Timedelta)
    as_period_series = pd.Series(as_period_index)
    check(assert_type(as_period_series, PeriodSeries), pd.Series, pd.Period)
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
    # https://github.com/pandas-dev/pandas/issues/50162
    check(assert_type(p + offset_index, pd.PeriodIndex), pd.Index)

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

    eq_s = check(
        assert_type(p == c_period_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(p != c_period_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
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

    eq_s = check(
        assert_type(c_period_series == p, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ne_s = check(
        assert_type(c_period_series != p, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (eq_s != ne_s).all()

    gt = check(assert_type(p > c_period, bool), bool)
    le = check(assert_type(p <= c_period, bool), bool)
    assert gt != le

    gt_a = check(assert_type(p > c_period_index, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(
        assert_type(p <= c_period_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_s = check(
        assert_type(p > c_period_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    le_s = check(
        assert_type(p <= c_period_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (gt_s != le_s).all()

    gt = check(assert_type(c_period > p, bool), bool)
    le = check(assert_type(c_period <= p, bool), bool)
    assert gt != le

    gt_a = check(assert_type(c_period_index > p, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(
        assert_type(c_period_index <= p, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (gt_a != le_a).all()

    gt_s = check(
        assert_type(c_period_series > p, "pd.Series[bool]"), pd.Series, np.bool_
    )
    le_s = check(
        assert_type(c_period_series <= p, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (gt_s != le_s).all()

    lt = check(assert_type(p < c_period, bool), bool)
    ge = check(assert_type(p >= c_period, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(p < c_period_index, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(
        assert_type(p >= c_period_index, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    lt_s = check(
        assert_type(p < c_period_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ge_s = check(
        assert_type(p >= c_period_series, "pd.Series[bool]"), pd.Series, np.bool_
    )
    assert (lt_s != ge_s).all()

    lt = check(assert_type(c_period < p, bool), bool)
    ge = check(assert_type(c_period >= p, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(c_period_index < p, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(
        assert_type(c_period_index >= p, np_ndarray_bool), np.ndarray, np.bool_
    )
    assert (lt_a != ge_a).all()

    lt_s = check(
        assert_type(c_period_series < p, "pd.Series[bool]"), pd.Series, np.bool_
    )
    ge_s = check(
        assert_type(c_period_series >= p, "pd.Series[bool]"), pd.Series, np.bool_
    )
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
