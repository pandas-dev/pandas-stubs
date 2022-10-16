from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    cast,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
import pytest
from typing_extensions import assert_type

from pandas._libs.tslibs.timedeltas import Components

if TYPE_CHECKING:
    from pandas.core.series import TimedeltaSeries  # noqa: F401
    from pandas.core.series import TimestampSeries  # noqa: F401

    from pandas._typing import np_ndarray_bool
else:
    TimedeltaSeries = TimestampSeries = np_ndarray_bool = Any

from tests import check

from pandas.tseries.offsets import Day


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
        assert_type(td + pd.TimedeltaIndex([td]), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(assert_type(td + pd.Series([td]), pd.Series), pd.Series)
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
    check(
        assert_type(pd.TimedeltaIndex([td]) + td, pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(assert_type(pd.Series([td]) + td, pd.Series), pd.Series)

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
    check(
        assert_type(td - pd.TimedeltaIndex([td]), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(assert_type(td - pd.Series([td]), pd.Series), pd.Series)
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
        assert_type(pd.TimedeltaIndex([td]) - td, pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.Series([td]) - td, Union[TimestampSeries, TimedeltaSeries]),
        pd.Series,
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
    check(assert_type(td * pd.Series([1, 2, 3]), TimedeltaSeries), pd.Series)
    check(assert_type(td * pd.Series([1.2, 2.2, 3.4]), TimedeltaSeries), pd.Series)
    check(assert_type(td * i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td * f_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(3 * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(3.5 * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Series([1, 2, 3]) * td, TimedeltaSeries), pd.Series)
    check(assert_type(pd.Series([1.2, 2.2, 3.4]) * td, TimedeltaSeries), pd.Series)
    check(assert_type(i_idx * td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(f_idx * td, pd.TimedeltaIndex), pd.TimedeltaIndex)

    np_intp_arr: npt.NDArray[np.integer] = np.array([1, 2, 3])
    np_float_arr: npt.NDArray[np.floating] = np.array([1, 2, 3])
    check(assert_type(td // td, int), int)
    check(assert_type(td // 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // 3.5, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // np_intp_arr, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td // np_float_arr, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td // pd.Series([1, 2, 3]), TimedeltaSeries), pd.Series)
    check(assert_type(td // pd.Series([1.2, 2.2, 3.4]), TimedeltaSeries), pd.Series)
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
    check(assert_type(td / pd.Series([1, 2, 3]), TimedeltaSeries), pd.Series)
    check(assert_type(td / pd.Series([1.2, 2.2, 3.4]), TimedeltaSeries), pd.Series)
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


def test_types_init() -> None:
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
    check(assert_type(s, TimestampSeries), pd.Series, pd.Timestamp)
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
