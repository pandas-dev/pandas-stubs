from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
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
    PeriodSeries: TypeAlias = pd.Series
    TimedeltaSeries: TypeAlias = pd.Series
    OffsetSeries: TypeAlias = pd.Series
    TimestampSeries: TypeAlias = pd.Series


def test_timedelta_construction() -> None:
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
    as1 = pd.Period("2012-01-01", freq="D")
    as2 = pd.Timestamp("2012-01-01")
    as3 = dt.datetime(2012, 1, 1)
    as4 = dt.date(2012, 1, 1)
    as5 = np.datetime64(1, "ns")
    as6 = dt.timedelta(days=1)
    as7 = np.timedelta64(1, "D")
    as8 = pd.TimedeltaIndex([td])
    as9 = pd.Series(as8)
    as10 = pd.period_range("2012-01-01", periods=3, freq="D")
    as11 = pd.date_range("2012-01-01", periods=3)
    as12 = ndarray_td64
    as13 = ndarray_dt64
    as14 = pd.NaT

    check(assert_type(td + td, pd.Timedelta), pd.Timedelta)
    check(assert_type(td + as1, pd.Period), pd.Period)
    check(assert_type(td + as2, pd.Timestamp), pd.Timestamp)
    check(assert_type(td + as3, pd.Timestamp), pd.Timestamp)
    check(assert_type(td + as4, dt.date), dt.date)
    check(assert_type(td + as5, pd.Timestamp), pd.Timestamp)
    check(assert_type(td + as6, pd.Timedelta), pd.Timedelta)
    check(assert_type(td + as7, pd.Timedelta), pd.Timedelta)
    check(assert_type(td + as8, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td + as9, TimedeltaSeries), pd.Series)
    check(assert_type(td + as10, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(td + as11, pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(td + as12, npt.NDArray[np.timedelta64]), np.ndarray, np.timedelta64
    )
    # pyright has trouble with timedelta64 and datetime64
    check(
        assert_type(
            td + as13,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
        np.datetime64,
    )
    check(assert_type(td + as14, NaTType), NaTType)

    check(assert_type(as1 + td, pd.Period), pd.Period)
    check(assert_type(as2 + td, pd.Timestamp), pd.Timestamp)
    check(assert_type(as3 + td, dt.datetime), dt.datetime)
    check(assert_type(as4 + td, dt.date), dt.date)
    check(assert_type(as5 + td, pd.Timestamp), pd.Timestamp)
    # pyright is wrong here because as6 + td calls td.__radd__(as6),
    # not timedelta.__add__
    check(
        assert_type(as6 + td, pd.Timedelta),  # pyright: ignore[reportGeneralTypeIssues]
        pd.Timedelta,
    )
    check(assert_type(as7 + td, pd.Timedelta), pd.Timedelta)
    check(assert_type(as8 + td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(as9 + td, TimedeltaSeries), pd.Series)
    check(assert_type(as10 + td, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(as11 + td, pd.DatetimeIndex), pd.DatetimeIndex)
    # pyright is wrong here because ndarray.__add__(Timedelta) is NotImplemented
    check(
        assert_type(
            as12 + td,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
        np.timedelta64,
    )
    check(assert_type(as14 + td, NaTType), NaTType)

    # sub is not symmetric with dates. In general date_like - timedelta is
    # sensible, while timedelta - date_like is not
    # TypeError: as1, as2, as3, as4, as5, as10, as11, as13
    check(assert_type(td - td, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as6, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as7, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as8, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td - as9, TimedeltaSeries), pd.Series)
    check(
        assert_type(td - as12, npt.NDArray[np.timedelta64]), np.ndarray, np.timedelta64
    )
    check(assert_type(td - as14, NaTType), NaTType)
    check(assert_type(as1 - td, pd.Period), pd.Period)
    check(assert_type(as2 - td, pd.Timestamp), pd.Timestamp)
    check(assert_type(as3 - td, dt.datetime), dt.datetime)
    check(assert_type(as4 - td, dt.date), dt.date)
    check(assert_type(as5 - td, pd.Timestamp), pd.Timestamp)
    # pyright is wrong here because as6 + td calls td.__rsub__(as6),
    # not timedelta.__sub__
    check(
        assert_type(
            as6 - td,  # pyright: ignore[reportGeneralTypeIssues]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(assert_type(as7 - td, pd.Timedelta), pd.Timedelta)
    check(assert_type(as8 - td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(as9 - td, TimedeltaSeries), pd.Series)
    check(assert_type(as10 - td, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(as11 - td, pd.DatetimeIndex), pd.DatetimeIndex)
    # pyright is wrong here because ndarray.__sub__(Timedelta) is NotImplemented
    check(
        assert_type(
            as12 - td,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.timedelta64],
        ),
        np.ndarray,
        np.timedelta64,
    )
    # pyright is wrong here because ndarray.__sub__(Timedelta) is NotImplemented
    check(
        assert_type(
            as13 - td,  # pyright: ignore[reportGeneralTypeIssues]
            npt.NDArray[np.datetime64],
        ),
        np.ndarray,
        np.datetime64,
    )
    check(assert_type(as14 - td, NaTType), NaTType)


def test_timedelta_mul_div() -> None:
    td = pd.Timedelta("1 day")

    with pytest.warns(FutureWarning):
        i_idx = cast(pd.Int64Index, pd.Index([1, 2, 3], dtype=int))
        f_idx = cast(pd.Float64Index, pd.Index([1.2, 2.2, 3.4], dtype=float))

    np_intp_arr: npt.NDArray[np.integer] = np.array([1, 2, 3])
    np_float_arr: npt.NDArray[np.floating] = np.array([1.2, 2.2, 3.4])

    md1 = 3
    md2 = 3.5
    md3 = np_intp_arr
    md4 = np_float_arr
    md5 = pd.Series([1, 2, 3])
    md6 = pd.Series([1.2, 2.2, 3.4])
    md7 = i_idx
    md8 = f_idx

    check(assert_type(td * md1, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * md2, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * md3, np.ndarray), np.ndarray)
    check(assert_type(td * md4, np.ndarray), np.ndarray)
    check(assert_type(td * md5, TimedeltaSeries), pd.Series)
    check(assert_type(td * md6, TimedeltaSeries), pd.Series)
    check(assert_type(td * md7, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td * md8, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(md1 * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(md2 * td, pd.Timedelta), pd.Timedelta)
    # pyright is wrong here ndarray.__mul__(Timedelta0 is NotImplemented
    check(
        assert_type(md3 * td, np.ndarray),  # pyright: ignore[reportGeneralTypeIssues]
        np.ndarray,
    )
    check(
        assert_type(md4 * td, np.ndarray),  # pyright: ignore[reportGeneralTypeIssues]
        np.ndarray,
    )
    check(assert_type(md5 * td, TimedeltaSeries), pd.Series)
    check(assert_type(md6 * td, TimedeltaSeries), pd.Series)
    check(assert_type(md7 * td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(md8 * td, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(td // td, int), int)
    check(assert_type(td // pd.NaT, float), float)
    check(assert_type(td // md1, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // md2, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // md3, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td // md4, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td // md5, TimedeltaSeries), pd.Series)
    check(assert_type(td // md6, TimedeltaSeries), pd.Series)
    check(assert_type(td // md7, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td // md8, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(pd.NaT // td, float), float)
    # Note: None of the reverse floordiv work
    # TypeError: md1, md2, md3, md4, md5, md6, md7, md8
    if TYPE_CHECKING_INVALID_USAGE:
        md1 // td  # type: ignore[operator]
        md2 // td  # type: ignore[operator]
        md3 // td  # type: ignore[operator]
        md4 // td  # type: ignore[operator]
        md5 // td  # type: ignore[operator]
        md6 // td  # type: ignore[operator]
        md7 // td  # type: ignore[operator]
        md8 // td  # type: ignore[operator]

    check(assert_type(td / td, float), float)
    check(assert_type(td / pd.NaT, float), float)
    check(assert_type(td / md1, pd.Timedelta), pd.Timedelta)
    check(assert_type(td / md2, pd.Timedelta), pd.Timedelta)
    check(assert_type(td / md3, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td / md4, npt.NDArray[np.timedelta64]), np.ndarray)
    check(assert_type(td / md5, TimedeltaSeries), pd.Series)
    check(assert_type(td / md6, TimedeltaSeries), pd.Series)
    check(assert_type(td / md7, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td / md8, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(pd.NaT / td, float), float)
    # Note: None of the reverse truediv work
    # TypeError: md1, md2, md3, md4, md5, md6, md7, md8
    if TYPE_CHECKING_INVALID_USAGE:
        md1 / td  # type: ignore[operator]
        md2 / td  # type: ignore[operator]
        md3 / td  # type: ignore[operator]
        md4 / td  # type: ignore[operator]
        # TODO: Series.__truediv__ says it supports Timedelta
        #   it does not, in general, except for TimedeltaSeries
        # md5 / td  # type: ignore[operator]
        # md6 / td  # type: ignore[operator]
        md7 / td  # type: ignore[operator]
        md8 / td  # type: ignore[operator]


def test_timedelta_mod_abs_unary() -> None:
    td = pd.Timedelta("1 day")

    with pytest.warns(FutureWarning):
        i_idx = cast(pd.Int64Index, pd.Index([1, 2, 3], dtype=int))
        f_idx = cast(pd.Float64Index, pd.Index([1.2, 2.2, 3.4], dtype=float))

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
    check(assert_type(td % int_series, TimedeltaSeries), pd.Series)
    check(assert_type(td % float_series, TimedeltaSeries), pd.Series)
    check(assert_type(td % i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(
        assert_type(td % f_idx, pd.TimedeltaIndex),
        pd.TimedeltaIndex,
    )

    # mypy and pyright reports dt.timedelta, even though __abs__ returns Timedelta
    check(assert_type(abs(td), dt.timedelta), pd.Timedelta)
    check(assert_type(td.__abs__(), pd.Timedelta), pd.Timedelta)
    check(assert_type(-td, pd.Timedelta), pd.Timedelta)
    check(assert_type(+td, pd.Timedelta), pd.Timedelta)


def test_timedelta_cmp() -> None:
    td = pd.Timedelta("1 day")
    ndarray_td64: npt.NDArray[np.timedelta64] = np.array(
        [1, 2, 3], dtype="timedelta64[D]"
    )
    c1 = td
    c2 = dt.timedelta(days=1)
    c3 = np.timedelta64(1, "D")
    c4 = ndarray_td64
    c5 = pd.TimedeltaIndex([1, 2, 3], unit="D")
    c6 = pd.Series([1, 2, 3], dtype="timedelta64[D]")

    check(assert_type(td < c1, bool), bool)
    check(assert_type(td < c2, bool), bool)
    check(assert_type(td < c3, bool), bool)
    check(assert_type(td < c4, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c5 < td, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c2 < td, bool), bool)
    check(assert_type(c4 < td, np_ndarray_bool), np.ndarray, np.bool_)
    check(assert_type(c5 < td, np_ndarray_bool), np.ndarray, np.bool_)

    gt = check(assert_type(td > c1, bool), bool)
    le = check(assert_type(td <= c1, bool), bool)
    assert gt != le

    gt = check(assert_type(td > c2, bool), bool)
    le = check(assert_type(td <= c2, bool), bool)
    assert gt != le

    gt = check(assert_type(td > c3, bool), bool)
    le = check(assert_type(td <= c3, bool), bool)
    assert gt != le

    gt_a = check(assert_type(td > c4, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(assert_type(td <= c4, np_ndarray_bool), np.ndarray, np.bool_)
    assert (gt_a != le_a).all()

    gt_a = check(assert_type(td > c5, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(assert_type(td <= c5, np_ndarray_bool), np.ndarray, np.bool_)
    assert (gt_a != le_a).all()

    gt_s = check(assert_type(td > c6, "pd.Series[bool]"), pd.Series, bool)
    le_s = check(assert_type(td <= c6, "pd.Series[bool]"), pd.Series, bool)
    assert (gt_s != le_s).all()

    gt = check(assert_type(c2 > td, bool), bool)
    le = check(assert_type(c2 <= td, bool), bool)
    assert gt != le

    gt_b = check(assert_type(c3 > td, Any), np.bool_)
    le_b = check(assert_type(c3 <= td, Any), np.bool_)
    assert gt_b != le_b

    gt_a = check(assert_type(c4 > td, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(assert_type(c4 <= td, np_ndarray_bool), np.ndarray, np.bool_)
    assert (gt_a != le_a).all()

    gt_a = check(assert_type(c5 > td, np_ndarray_bool), np.ndarray, np.bool_)
    le_a = check(assert_type(c5 <= td, np_ndarray_bool), np.ndarray, np.bool_)
    assert (gt_a != le_a).all()

    eq_s = check(assert_type(c6 > td, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(c6 <= td, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    lt = check(assert_type(td < c1, bool), bool)
    ge = check(assert_type(td >= c1, bool), bool)
    assert lt != ge

    lt = check(assert_type(td < c2, bool), bool)
    ge = check(assert_type(td >= c2, bool), bool)
    assert lt != ge

    lt = check(assert_type(td < c3, bool), bool)
    ge = check(assert_type(td >= c3, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(td < c4, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(assert_type(td >= c4, np_ndarray_bool), np.ndarray, np.bool_)
    assert (lt_a != ge_a).all()

    lt_a = check(assert_type(td < c5, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(assert_type(td >= c5, np_ndarray_bool), np.ndarray, np.bool_)
    assert (lt_a != ge_a).all()

    eq_s = check(assert_type(td < c6, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(td >= c6, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    lt = check(assert_type(c2 < td, bool), bool)
    ge = check(assert_type(c2 >= td, bool), bool)
    assert lt != ge

    lt_b = check(assert_type(c3 < td, Any), np.bool_)
    ge_b = check(assert_type(c3 >= td, Any), np.bool_)
    assert lt_b != ge_b

    lt_a = check(assert_type(c4 < td, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(assert_type(c4 >= td, np_ndarray_bool), np.ndarray, np.bool_)
    assert (lt_a != ge_a).all()

    lt_a = check(assert_type(c5 < td, np_ndarray_bool), np.ndarray, np.bool_)
    ge_a = check(assert_type(c5 >= td, np_ndarray_bool), np.ndarray, np.bool_)
    assert (lt_a != ge_a).all()

    eq_s = check(assert_type(c6 < td, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(c6 >= td, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    eq = check(assert_type(td == td, bool), bool)
    ne = check(assert_type(td != td, bool), bool)
    assert eq != ne

    eq = check(assert_type(td == c2, bool), bool)
    ne = check(assert_type(td != c2, bool), bool)
    assert eq != ne

    eq = check(assert_type(td == c3, bool), bool)
    ne = check(assert_type(td != c3, bool), bool)
    assert eq != ne

    eq_a = check(assert_type(td == c4, np_ndarray_bool), np.ndarray, np.bool_)
    ne_a = check(assert_type(td != c4, np_ndarray_bool), np.ndarray, np.bool_)
    assert (eq_a != ne_a).all()

    eq_a = check(assert_type(td == c5, np_ndarray_bool), np.ndarray, np.bool_)
    ne_a = check(assert_type(td != c5, np_ndarray_bool), np.ndarray, np.bool_)
    assert (eq_a != ne_a).all()

    eq_s = check(assert_type(td == c6, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(td != c6, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    eq = check(assert_type(c2 == td, bool), bool)
    ne = check(assert_type(c2 != td, bool), bool)
    assert eq != ne

    eq = check(assert_type(c3 == td, Any), np.bool_)
    ne = check(assert_type(c3 != td, Any), np.bool_)
    assert eq != ne

    eq_a = check(assert_type(c4 == td, Any), np.ndarray)
    ne_a = check(assert_type(c4 != td, Any), np.ndarray)
    assert (eq_a != ne_a).all()

    eq_a = check(assert_type(c5 == td, np_ndarray_bool), np.ndarray, np.bool_)
    ne_a = check(assert_type(c5 != td, np_ndarray_bool), np.ndarray, np.bool_)
    assert (eq_a != ne_a).all()

    eq_s = check(assert_type(c6 == td, "pd.Series[bool]"), pd.Series, bool)
    ne_s = check(assert_type(c6 != td, "pd.Series[bool]"), pd.Series, bool)
    assert (eq_s != ne_s).all()

    eq = check(assert_type(td == 1, bool), bool)
    ne = check(assert_type(td != 1, bool), bool)
    assert eq != ne

    eq = check(assert_type(td == (3 + 2j), bool), bool)
    ne = check(assert_type(td != (3 + 2j), bool), bool)
    assert eq != ne

    eq_s = check(
        assert_type(td == pd.Series([1, 2, 3]), "pd.Series[bool]"), pd.Series, bool
    )
    ne_s = check(
        assert_type(td != pd.Series([1, 2, 3]), "pd.Series[bool]"), pd.Series, bool
    )
    assert (eq_s != ne_s).all()


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
