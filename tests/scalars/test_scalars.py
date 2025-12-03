from __future__ import annotations

import datetime
import datetime as dt
from typing import (
    Any,
    Literal,
    TypeAlias,
)

import dateutil.tz
import numpy as np
import pandas as pd
from pandas.api.typing import NaTType
import pytz
from typing_extensions import assert_type

from pandas._libs.tslibs.timedeltas import Components
from pandas._typing import TimeUnit

from tests import (
    PD_LTE_23,
    TYPE_CHECKING_INVALID_USAGE,
    check,
    pytest_warns_bounded,
)
from tests._typing import (
    np_1darray_bool,
    np_2darray,
    np_ndarray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_td,
)

from pandas.tseries.offsets import (
    BaseOffset,
    Day,
)

if not PD_LTE_23:
    from pandas.errors import Pandas4Warning  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue,reportRedeclaration,reportUnknownVariableType] # isort: skip
else:
    Pandas4Warning: TypeAlias = FutureWarning  # type: ignore[no-redef]


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

    if TYPE_CHECKING_INVALID_USAGE:
        _i = interval_i - pd.Interval(1, 2)  # type: ignore[type-var] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _f = interval_f - pd.Interval(1.0, 2.0)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        # TODO: psf/black#4880
        # fmt: off
        _ts = (  # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
            interval_ts
            - pd.Interval(  # type: ignore[operator]
                pd.Timestamp(2025, 9, 29), pd.Timestamp(2025, 9, 30), closed="both"
            )
        )
        _td = (  # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
            interval_td
            - pd.Interval(  # type: ignore[operator]
                pd.Timedelta(1, "ns"), pd.Timedelta(2, "ns")
            )
        )
        # fmt: on


def test_interval_cmp() -> None:
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
        assert_type(interval_index_int >= interval_i, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_index_int < interval_i, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_index_int <= interval_i, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_index_int > interval_i, np_1darray_bool),
        np_1darray_bool,
    )

    check(
        assert_type(interval_i >= interval_index_int, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_i < interval_index_int, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_i <= interval_index_int, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_i > interval_index_int, np_1darray_bool),
        np_1darray_bool,
    )

    check(
        assert_type(interval_index_int == interval_i, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_index_int != interval_i, np_1darray_bool),
        np_1darray_bool,
    )

    check(
        assert_type(interval_i == interval_index_int, np_1darray_bool),
        np_1darray_bool,
    )
    check(
        assert_type(interval_i != interval_index_int, np_1darray_bool),
        np_1darray_bool,
    )


def test_timedelta_construction() -> None:
    check(assert_type(pd.Timedelta(1, "W"), pd.Timedelta), pd.Timedelta)
    with pytest_warns_bounded(
        Pandas4Warning,  # should be Pandas4Warning but only exposed starting pandas 3.0.0
        "'w' is deprecated and will",
        lower="2.3.99",
        upper="3.0.99",
    ):
        check(assert_type(pd.Timedelta(1, "w"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "D"), pd.Timedelta), pd.Timedelta)
    with pytest_warns_bounded(Pandas4Warning, "'d' is deprecated", lower="2.3.99"):
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
    check(assert_type(pd.Timedelta(1, "s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "seconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "sec"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "second"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "ms"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "milliseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "millisecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "milli"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "millis"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "us"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "microseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "microsecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "µs"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "micro"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "micros"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "ns"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nanoseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nano"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nanos"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta(1, "nanosecond"), pd.Timedelta), pd.Timedelta)

    check(assert_type(pd.Timedelta("1 W"), pd.Timedelta), pd.Timedelta)
    with pytest_warns_bounded(Pandas4Warning, "'w' is deprecated", lower="2.3.99"):
        check(assert_type(pd.Timedelta("1 w"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 D"), pd.Timedelta), pd.Timedelta)
    with pytest_warns_bounded(Pandas4Warning, "'d' is deprecated", lower="2.3.99"):
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
    check(assert_type(pd.Timedelta("1 s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 seconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 sec"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 second"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 ms"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 milliseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 millisecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 milli"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 millis"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 us"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 microseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 microsecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 µs"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 micro"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 micros"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 ns"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nanoseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nano"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nanos"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.Timedelta("1 nanosecond"), pd.Timedelta), pd.Timedelta)
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
    check(assert_type(td.unit, TimeUnit), str)

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

    check(assert_type(td.as_unit("s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(td.as_unit("ms"), pd.Timedelta), pd.Timedelta)
    check(assert_type(td.as_unit("us", round_ok=True), pd.Timedelta), pd.Timedelta)
    check(assert_type(td.as_unit("ns", round_ok=False), pd.Timedelta), pd.Timedelta)


def test_timedelta_add_sub() -> None:
    td = pd.Timedelta("1 day")

    ndarray_td64: np_ndarray_td = np.array([1, 2, 3], dtype="timedelta64[D]")
    ndarray_dt64: np_ndarray_dt = np.array([1, 2, 3], dtype="datetime64[D]")
    as_period = pd.Period("2012-01-01", freq="D")
    as_timestamp = pd.Timestamp("2012-01-01")
    as_datetime = dt.datetime(2012, 1, 1)
    as_date = dt.date(2012, 1, 1)
    as_datetime64 = np.datetime64(1, "ns")
    as_dt_timedelta = dt.timedelta(days=1)
    as_timedelta64 = np.timedelta64(1, "D")
    as_timedelta_index = pd.TimedeltaIndex([td])
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
    check(assert_type(td + as_period_index, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(td + as_datetime_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(td + as_ndarray_td64, np_ndarray_td), np_ndarray, np.timedelta64)
    check(assert_type(td + as_ndarray_dt64, np_ndarray_dt), np_ndarray, np.datetime64)
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
            as_dt_timedelta + td,  # pyright: ignore[reportAssertTypeFailure]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(
        assert_type(  # type: ignore[assert-type]
            as_timedelta64 + td,  # pyright: ignore[reportAssertTypeFailure]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(assert_type(as_timedelta_index + td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(as_period_index + td, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(as_datetime_index + td, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(as_ndarray_td64 + td, np_ndarray_td), np_ndarray, np.timedelta64)
    check(assert_type(as_nat + td, NaTType), NaTType)

    # sub is not symmetric with dates. In general date_like - timedelta is
    # sensible, while timedelta - date_like is not
    # TypeError: as_period, as_timestamp, as_datetime, as_date, as_datetime64,
    #            as_period_index, as_datetime_index, as_ndarray_dt64
    if TYPE_CHECKING_INVALID_USAGE:
        _0 = td - as_period  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _1 = td - as_timestamp  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _2 = td - as_datetime  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _3 = td - as_date  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _4 = td - as_datetime64  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _5 = td - as_period_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _6 = td - as_datetime_index  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _7 = td - as_ndarray_dt64  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    check(assert_type(td - td, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as_dt_timedelta, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as_timedelta64, pd.Timedelta), pd.Timedelta)
    check(assert_type(td - as_timedelta_index, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(td - as_ndarray_td64, np_ndarray_td), np_ndarray, np.timedelta64)
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
            as_dt_timedelta - td,  # pyright: ignore[reportAssertTypeFailure]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(
        assert_type(  # type: ignore[assert-type]
            as_timedelta64 - td,  # pyright: ignore[reportAssertTypeFailure]
            pd.Timedelta,
        ),
        pd.Timedelta,
    )
    check(assert_type(as_timedelta_index - td, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(as_period_index - td, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(as_datetime_index - td, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(as_ndarray_td64 - td, np_ndarray_td), np_ndarray, np.timedelta64)
    check(assert_type(as_ndarray_dt64 - td, np_ndarray_dt), np_ndarray, np.datetime64)
    check(assert_type(as_nat - td, NaTType), NaTType)


def test_timedelta_mul_div() -> None:
    td = pd.Timedelta("1 day")

    np_intp_arr: np_ndarray_anyint = np.array([1, 2, 3])
    np_float_arr: np_ndarray_float = np.array([1.2, 2.2, 3.4])

    md_int = 3
    md_float = 3.5
    md_ndarray_intp = np_intp_arr
    md_ndarray_float = np_float_arr

    check(assert_type(td * md_int, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * md_float, pd.Timedelta), pd.Timedelta)
    check(assert_type(td * md_ndarray_intp, np_ndarray_td), np_ndarray, np.timedelta64)
    check(assert_type(td * md_ndarray_float, np_ndarray_td), np_ndarray, np.timedelta64)

    check(assert_type(md_int * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(md_float * td, pd.Timedelta), pd.Timedelta)
    check(assert_type(md_ndarray_intp * td, np_ndarray_td), np_ndarray, np.timedelta64)
    check(assert_type(md_ndarray_float * td, np_ndarray_td), np_ndarray, np.timedelta64)

    check(assert_type(td // td, int), int)
    check(assert_type(td // pd.NaT, float), float)
    check(assert_type(td // md_int, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // md_float, pd.Timedelta), pd.Timedelta)
    check(assert_type(td // md_ndarray_intp, np_ndarray_td), np_ndarray, np.timedelta64)
    check(
        assert_type(td // md_ndarray_float, np_ndarray_td), np_ndarray, np.timedelta64
    )

    check(assert_type(pd.NaT // td, float), float)
    # Note: None of the reverse floordiv work
    # TypeError: md_int, md_float, md_ndarray_intp, md_ndarray_float, mp_series_int,
    #            mp_series_float, md_int64_index, md_float_index
    if TYPE_CHECKING_INVALID_USAGE:
        _00 = md_int // td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _01 = md_float // td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _02 = md_ndarray_intp // td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _03 = md_ndarray_float // td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    check(assert_type(td / td, float), float)
    check(assert_type(td / pd.NaT, float), float)
    check(assert_type(td / md_int, pd.Timedelta), pd.Timedelta)
    check(assert_type(td / md_float, pd.Timedelta), pd.Timedelta)
    check(assert_type(td / md_ndarray_intp, np_ndarray_td), np_ndarray, np.timedelta64)
    check(assert_type(td / md_ndarray_float, np_ndarray_td), np_ndarray, np.timedelta64)

    check(assert_type(pd.NaT / td, float), float)
    # Note: None of the reverse truediv work
    # TypeError: md_int, md_float, md_ndarray_intp, md_ndarray_float, mp_series_int,
    #            mp_series_float, md_int64_index, md_float_index
    if TYPE_CHECKING_INVALID_USAGE:
        _10 = md_int / td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _11 = md_float / td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _12 = md_ndarray_intp / td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _13 = md_ndarray_float / td  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]


def test_timedelta_mod_abs_unary() -> None:
    td = pd.Timedelta("1 day")

    i_idx = pd.Index([1, 2, 3], dtype=int)
    f_idx = pd.Index([1.2, 2.2, 3.4], dtype=float)

    check(assert_type(td % 3, pd.Timedelta), pd.Timedelta)
    check(assert_type(td % 3.5, pd.Timedelta), pd.Timedelta)
    check(assert_type(td % td, pd.Timedelta), pd.Timedelta)
    check(
        assert_type(td % np.array([1, 2, 3]), np_ndarray_td), np_ndarray, np.timedelta64
    )
    check(
        assert_type(td % np.array([1.2, 2.2, 3.4]), np_ndarray_td),
        np_ndarray,
        np.timedelta64,
    )
    int_series = pd.Series([1, 2, 3], dtype=int)
    float_series = pd.Series([1.2, 2.2, 3.4], dtype=float)
    check(
        assert_type(td % int_series, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta
    )
    check(
        assert_type(td % float_series, "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
    check(assert_type(td % i_idx, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(
        assert_type(td % f_idx, pd.TimedeltaIndex),
        pd.TimedeltaIndex,
    )

    check(assert_type(abs(td), pd.Timedelta), pd.Timedelta)

    check(assert_type(td.__abs__(), pd.Timedelta), pd.Timedelta)
    check(assert_type(-td, pd.Timedelta), pd.Timedelta)
    check(assert_type(+td, pd.Timedelta), pd.Timedelta)


def test_timedelta_cmp_scalar() -> None:
    td = pd.Timedelta("1 day")
    td2 = pd.Timedelta("1 hour")
    py_td = dt.timedelta(days=1)
    np_td = np.timedelta64(1, "D")

    # >, <=
    gt1 = check(assert_type(td > td2, bool), bool)
    le1 = check(assert_type(td <= td2, bool), bool)
    assert gt1 != le1
    gt2 = check(assert_type(td > py_td, bool), bool)
    le2 = check(assert_type(td <= py_td, bool), bool)
    assert gt2 != le2
    gt3 = check(assert_type(td > np_td, bool), bool)
    le3 = check(assert_type(td <= np_td, bool), bool)
    assert gt3 != le3
    gt4 = check(assert_type(py_td > td, bool), bool)
    le4 = check(assert_type(py_td <= td, bool), bool)
    assert gt4 != le4
    gt5 = check(assert_type(np_td > td, np.bool), bool)
    le5 = check(assert_type(np_td <= td, np.bool), bool)
    assert gt5 != le5

    # <, >=
    lt1 = check(assert_type(td < td2, bool), bool)
    ge1 = check(assert_type(td >= td2, bool), bool)
    assert lt1 != ge1
    lt2 = check(assert_type(td < py_td, bool), bool)
    ge2 = check(assert_type(td >= py_td, bool), bool)
    assert lt2 != ge2
    lt3 = check(assert_type(td < np_td, bool), bool)
    ge3 = check(assert_type(td >= np_td, bool), bool)
    assert lt3 != ge3
    lt4 = check(assert_type(py_td < td, bool), bool)
    ge4 = check(assert_type(py_td >= td, bool), bool)
    assert lt4 != ge4
    lt5 = check(assert_type(np_td < td, np.bool), bool)
    ge5 = check(assert_type(np_td >= td, np.bool), bool)
    assert lt5 != ge5

    # ==, !=
    eq1 = check(assert_type(td == td, bool), bool)
    ne1 = check(assert_type(td != td, bool), bool)
    assert eq1 != ne1
    eq2 = check(assert_type(td == py_td, bool), bool)
    ne2 = check(assert_type(td != py_td, bool), bool)
    assert eq2 != ne2
    eq3 = check(assert_type(td == np_td, bool), bool)
    ne3 = check(assert_type(td != np_td, bool), bool)
    assert eq3 != ne3
    eq4 = check(assert_type(td == 1, Literal[False]), bool)
    ne4 = check(assert_type(td != 1, Literal[True]), bool)
    assert eq4 != ne4
    eq5 = check(assert_type(td == (3 + 2j), Literal[False]), bool)
    ne5 = check(assert_type(td != (3 + 2j), Literal[True]), bool)
    assert eq5 != ne5

    # ==, != (td on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(py_td == td, bool), bool)
    ne_rhs1 = check(assert_type(py_td != td, bool), bool)
    assert eq_rhs1 != ne_rhs1
    eq_rhs2 = check(assert_type(np_td == td, Any), bool)
    ne_rhs2 = check(assert_type(np_td != td, Any), bool)
    assert eq_rhs2 != ne_rhs2


def test_timedelta_cmp_series() -> None:
    td = pd.Timedelta("1 day")
    td_ser = pd.Series(pd.TimedeltaIndex([1, 2, 3]))  # TimedeltaSeries

    # >, <=
    gt1 = check(assert_type(td > td_ser, "pd.Series[bool]"), pd.Series, np.bool)
    le1 = check(assert_type(td <= td_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (gt1 != le1).all()
    gt2 = check(assert_type(td_ser > td, "pd.Series[bool]"), pd.Series, np.bool)
    le2 = check(assert_type(td_ser <= td, "pd.Series[bool]"), pd.Series, np.bool)
    assert (gt2 != le2).all()

    # <, >=
    lt1 = check(assert_type(td < td_ser, "pd.Series[bool]"), pd.Series, np.bool)
    ge1 = check(assert_type(td >= td_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (lt1 != ge1).all()
    lt2 = check(assert_type(td_ser < td, "pd.Series[bool]"), pd.Series, np.bool)
    ge2 = check(assert_type(td_ser >= td, "pd.Series[bool]"), pd.Series, np.bool)
    assert (lt2 != ge2).all()

    # ==, !=
    eq1 = check(assert_type(td == td_ser, "pd.Series[bool]"), pd.Series, np.bool)
    ne1 = check(assert_type(td != td_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (eq1 != ne1).all()

    # ==, != (td on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(td_ser == td, "pd.Series[bool]"), pd.Series, np.bool)
    ne_rhs1 = check(assert_type(td_ser != td, "pd.Series[bool]"), pd.Series, np.bool)
    assert (eq_rhs1 != ne_rhs1).all()


def test_timedelta_cmp_index() -> None:
    td = pd.Timedelta("1 day")
    td_idx = pd.to_timedelta([1, 2, 3], unit="D")  # TimedeltaIndex

    # >, <=
    gt1 = check(assert_type(td > td_idx, np_1darray_bool), np_1darray_bool)
    le1 = check(assert_type(td <= td_idx, np_1darray_bool), np_1darray_bool)
    assert (gt1 != le1).all()
    gt2 = check(assert_type(td_idx > td, np_1darray_bool), np_1darray_bool)
    le2 = check(assert_type(td_idx <= td, np_1darray_bool), np_1darray_bool)
    assert (gt2 != le2).all()

    # <, >=
    lt1 = check(assert_type(td < td_idx, np_1darray_bool), np_1darray_bool)
    ge1 = check(assert_type(td >= td_idx, np_1darray_bool), np_1darray_bool)
    assert (lt1 != ge1).all()
    lt2 = check(assert_type(td_idx < td, np_1darray_bool), np_1darray_bool)
    ge2 = check(assert_type(td_idx >= td, np_1darray_bool), np_1darray_bool)
    assert (lt2 != ge2).all()

    # ==, !=
    eq1 = check(assert_type(td == td_idx, np_1darray_bool), np_1darray_bool)
    ne1 = check(assert_type(td != td_idx, np_1darray_bool), np_1darray_bool)
    assert (eq1 != ne1).all()

    # ==, != (td on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(td_idx == td, np_1darray_bool), np_1darray_bool)
    ne_rhs1 = check(assert_type(td_idx != td, np_1darray_bool), np_1darray_bool)
    assert (eq_rhs1 != ne_rhs1).all()


def test_timedelta_cmp_array() -> None:
    td = pd.Timedelta("1 day")
    arr_1d = pd.to_timedelta([1, 2, 3, 4], unit="D").to_numpy()
    arr_2d = arr_1d.reshape(2, 2)
    arr_nd = arr_1d.reshape([4])

    # >, <=
    gt_nd1 = check(assert_type(td > arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    le_nd1 = check(assert_type(td <= arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (gt_nd1 != le_nd1).all()
    gt_nd2 = check(assert_type(arr_nd > td, np_ndarray_bool), np_ndarray_bool, np.bool)
    le_nd2 = check(assert_type(arr_nd <= td, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (gt_nd2 != le_nd2).all()
    gt_2d1 = check(assert_type(td > arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    le_2d1 = check(assert_type(td <= arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (gt_2d1 != le_2d1).all()
    gt_2d2 = check(assert_type(arr_2d > td, np_2darray[np.bool]), np_2darray[np.bool])
    le_2d2 = check(assert_type(arr_2d <= td, np_2darray[np.bool]), np_2darray[np.bool])
    assert (gt_2d2 != le_2d2).all()
    gt_1d1 = check(assert_type(td > arr_1d, np_1darray_bool), np_1darray_bool)
    le_1d1 = check(assert_type(td <= arr_1d, np_1darray_bool), np_1darray_bool)
    assert (gt_1d1 != le_1d1).all()
    gt_1d2 = check(assert_type(arr_1d > td, np_1darray_bool), np_1darray_bool)
    le_1d2 = check(assert_type(arr_1d <= td, np_1darray_bool), np_1darray_bool)
    assert (gt_1d2 != le_1d2).all()

    # <, >=
    lt_nd1 = check(assert_type(td < arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    ge_nd1 = check(assert_type(td >= arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (lt_nd1 != ge_nd1).all()
    lt_nd2 = check(assert_type(arr_nd < td, np_ndarray_bool), np_ndarray_bool, np.bool)
    ge_nd2 = check(assert_type(arr_nd >= td, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (lt_nd2 != ge_nd2).all()
    lt_2d1 = check(assert_type(td < arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    ge_2d1 = check(assert_type(td >= arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (lt_2d1 != ge_2d1).all()
    lt_2d2 = check(assert_type(arr_2d < td, np_2darray[np.bool]), np_2darray[np.bool])
    ge_2d2 = check(assert_type(arr_2d >= td, np_2darray[np.bool]), np_2darray[np.bool])
    assert (lt_2d2 != ge_2d2).all()
    lt_1d1 = check(assert_type(td < arr_1d, np_1darray_bool), np_1darray_bool)
    ge_1d1 = check(assert_type(td >= arr_1d, np_1darray_bool), np_1darray_bool)
    assert (lt_1d1 != ge_1d1).all()
    lt_1d2 = check(assert_type(arr_1d < td, np_1darray_bool), np_1darray_bool)
    ge_1d2 = check(assert_type(arr_1d >= td, np_1darray_bool), np_1darray_bool)
    assert (lt_1d2 != ge_1d2).all()

    # ==, !=
    eq_nd1 = check(assert_type(td == arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    ne_nd1 = check(assert_type(td != arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (eq_nd1 != ne_nd1).all()
    eq_2d1 = check(assert_type(td == arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    ne_2d1 = check(assert_type(td != arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (eq_2d1 != ne_2d1).all()
    eq_1d1 = check(assert_type(td == arr_1d, np_1darray_bool), np_1darray_bool)
    ne_1d1 = check(assert_type(td != arr_1d, np_1darray_bool), np_1darray_bool)
    assert (eq_1d1 != ne_1d1).all()

    # ==, != (td on the rhs, use == and != of lhs)
    eq_rhs_nd1 = check(assert_type(arr_nd == td, Any), np_ndarray_bool)
    ne_rhs_nd1 = check(assert_type(arr_nd != td, Any), np_ndarray_bool)
    assert (eq_rhs_nd1 != ne_rhs_nd1).all()
    eq_rhs_2d1 = check(assert_type(arr_2d == td, Any), np_2darray[np.bool])
    ne_rhs_2d1 = check(assert_type(arr_2d != td, Any), np_2darray[np.bool])
    assert (eq_rhs_2d1 != ne_rhs_2d1).all()
    eq_rhs_1d1 = check(assert_type(arr_1d == td, Any), np_1darray_bool)
    ne_rhs_1d1 = check(assert_type(arr_1d != td, Any), np_1darray_bool)
    assert (eq_rhs_1d1 != ne_rhs_1d1).all()


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
    check(assert_type(ts.tz, dt.tzinfo | None), type(None))
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
    check(assert_type(ts.tzinfo, dt.tzinfo | None), type(None))
    check(assert_type(ts.value, int), int)
    check(assert_type(ts.year, int), int)
    check(assert_type(ts.unit, TimeUnit), str)


def test_timestamp_add_sub() -> None:
    ts = pd.Timestamp("2000-1-1")
    np_td64_arr: np_ndarray_td = np.array([1, 2], dtype="timedelta64[ns]")

    as_pd_timedelta = pd.Timedelta(days=1)
    as_dt_timedelta = dt.timedelta(days=1)
    as_offset = 3 * Day()

    as_timedelta_index = pd.to_timedelta([1, 2, 3], unit="D")
    as_timedelta_series = pd.Series(as_timedelta_index)
    check(
        assert_type(as_timedelta_series, "pd.Series[pd.Timedelta]"),
        pd.Series,
        pd.Timedelta,
    )
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
        assert_type(ts + as_timedelta_series, "pd.Series[pd.Timestamp]"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(as_timedelta_series + ts, "pd.Series[pd.Timestamp]"),
        pd.Series,
        pd.Timestamp,
    )

    check(
        assert_type(ts + as_np_ndarray_td64, np_ndarray_dt), np_ndarray, np.datetime64
    )
    check(
        assert_type(as_np_ndarray_td64 + ts, np_ndarray_dt), np_ndarray, np.datetime64
    )

    # Reverse order is not possible for all of these
    check(assert_type(ts - as_pd_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_dt_timedelta, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_offset, pd.Timestamp), pd.Timestamp)
    check(assert_type(ts - as_timedelta_index, pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(ts - as_timedelta_series, "pd.Series[pd.Timestamp]"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(ts - as_np_ndarray_td64, np_ndarray_dt), np_ndarray, np.datetime64
    )


def test_timestamp_cmp_scalar() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)
    ts2 = pd.Timestamp(year=2025, month=8, day=16)
    py_dt = dt.datetime(year=2000, month=1, day=1)
    np_dt = np.datetime64(1, "ns")

    # >, <=
    gt1 = check(assert_type(ts > ts2, bool), bool)
    le1 = check(assert_type(ts <= ts2, bool), bool)
    assert gt1 != le1
    gt2 = check(assert_type(ts > py_dt, bool), bool)
    le2 = check(assert_type(ts <= py_dt, bool), bool)
    assert gt2 != le2
    gt3 = check(assert_type(ts > np_dt, bool), bool)
    le3 = check(assert_type(ts <= np_dt, bool), bool)
    assert gt3 != le3
    gt4 = check(assert_type(py_dt > ts, bool), bool)
    le4 = check(assert_type(py_dt <= ts, bool), bool)
    assert gt4 != le4
    gt5 = check(assert_type(np_dt > ts, np.bool), bool)
    le5 = check(assert_type(np_dt <= ts, np.bool), bool)
    assert gt5 != le5

    # <, >=
    lt1 = check(assert_type(ts < ts2, bool), bool)
    ge1 = check(assert_type(ts >= ts2, bool), bool)
    assert ge1 != lt1
    lt2 = check(assert_type(ts < py_dt, bool), bool)
    ge2 = check(assert_type(ts >= py_dt, bool), bool)
    assert ge2 != lt2
    lt3 = check(assert_type(ts < np_dt, bool), bool)
    ge3 = check(assert_type(ts >= np_dt, bool), bool)
    assert ge3 != lt3
    lt4 = check(assert_type(py_dt < ts, bool), bool)
    ge4 = check(assert_type(py_dt >= ts, bool), bool)
    assert ge4 != lt4
    lt5 = check(assert_type(np_dt < ts, np.bool), bool)
    ge5 = check(assert_type(np_dt >= ts, np.bool), bool)
    assert ge5 != lt5

    # =, !=
    eq1 = check(assert_type(ts == ts2, bool), bool)
    ne1 = check(assert_type(ts != ts2, bool), bool)
    assert eq1 != ne1
    eq2 = check(assert_type(ts == py_dt, bool), bool)
    ne2 = check(assert_type(ts != py_dt, bool), bool)
    assert eq2 != ne2
    eq3 = check(assert_type(ts == np_dt, bool), bool)
    ne3 = check(assert_type(ts != np_dt, bool), bool)
    assert eq3 != ne3
    eq4 = check(assert_type(ts == 1, Literal[False]), bool)
    ne4 = check(assert_type(ts != 1, Literal[True]), bool)
    assert eq4 != ne4
    eq5 = check(assert_type(ts == (3 + 2j), Literal[False]), bool)
    ne5 = check(assert_type(ts != (3 + 2j), Literal[True]), bool)
    assert eq5 != ne5

    # ==, != (ts on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(py_dt == ts, bool), bool)
    ne_rhs1 = check(assert_type(py_dt != ts, bool), bool)
    assert eq_rhs1 != ne_rhs1
    eq_rhs2 = check(assert_type(np_dt == ts, Any), bool)
    ne_rhs2 = check(assert_type(np_dt != ts, Any), bool)
    assert eq_rhs2 != ne_rhs2


def test_timestamp_cmp_series() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)
    ts_ser = pd.Series(pd.DatetimeIndex(["2000-1-1", "2000-1-2"]))
    check(assert_type(ts_ser, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)

    # >, <=
    gt1 = check(assert_type(ts > ts_ser, "pd.Series[bool]"), pd.Series, np.bool)
    le1 = check(assert_type(ts <= ts_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (gt1 != le1).all()
    gt2 = check(assert_type(ts_ser > ts, "pd.Series[bool]"), pd.Series, np.bool)
    le2 = check(assert_type(ts_ser <= ts, "pd.Series[bool]"), pd.Series, np.bool)
    assert (gt2 != le2).all()

    # <, >=
    lt1 = check(assert_type(ts < ts_ser, "pd.Series[bool]"), pd.Series, np.bool)
    ge1 = check(assert_type(ts >= ts_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (lt1 != ge1).all()
    lt2 = check(assert_type(ts_ser < ts, "pd.Series[bool]"), pd.Series, np.bool)
    ge2 = check(assert_type(ts_ser >= ts, "pd.Series[bool]"), pd.Series, np.bool)
    assert (lt2 != ge2).all()

    # ==, !=
    eq1 = check(assert_type(ts == ts_ser, "pd.Series[bool]"), pd.Series, np.bool)
    ne1 = check(assert_type(ts != ts_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (eq1 != ne1).all()

    # ==, != (ts on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(ts_ser == ts, "pd.Series[bool]"), pd.Series, np.bool)
    ne_rhs1 = check(assert_type(ts_ser != ts, "pd.Series[bool]"), pd.Series, np.bool)
    assert (eq_rhs1 != ne_rhs1).all()


def test_timestamp_cmp_index() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)
    dt_idx = pd.DatetimeIndex(["2000-1-1"])
    # DatetimeIndex, but the type checker thinks it is Index[Any].
    un_idx = pd.DataFrame({"a": [1]}, index=dt_idx).index

    # >, <=
    gt_dt1 = check(assert_type(ts > dt_idx, np_1darray_bool), np_1darray_bool)
    le_dt1 = check(assert_type(ts <= dt_idx, np_1darray_bool), np_1darray_bool)
    assert (gt_dt1 != le_dt1).all()
    gt_dt2 = check(assert_type(dt_idx > ts, np_1darray_bool), np_1darray_bool)
    le_dt2 = check(assert_type(dt_idx <= ts, np_1darray_bool), np_1darray_bool)
    assert (gt_dt2 != le_dt2).all()
    gt_un1 = check(assert_type(ts > un_idx, np_1darray_bool), np_1darray_bool)
    le_un1 = check(assert_type(ts <= un_idx, np_1darray_bool), np_1darray_bool)
    assert (gt_un1 != le_un1).all()
    gt_un2 = check(assert_type(un_idx > ts, np_1darray_bool), np_1darray_bool)
    le_un2 = check(assert_type(un_idx <= ts, np_1darray_bool), np_1darray_bool)
    assert (gt_un2 != le_un2).all()

    # <, >=
    lt_dt1 = check(assert_type(ts < dt_idx, np_1darray_bool), np_1darray_bool)
    ge_dt1 = check(assert_type(ts >= dt_idx, np_1darray_bool), np_1darray_bool)
    assert (lt_dt1 != ge_dt1).all()
    lt_dt2 = check(assert_type(dt_idx < ts, np_1darray_bool), np_1darray_bool)
    ge_dt2 = check(assert_type(dt_idx >= ts, np_1darray_bool), np_1darray_bool)
    assert (lt_dt2 != ge_dt2).all()
    lt_un1 = check(assert_type(ts < un_idx, np_1darray_bool), np_1darray_bool)
    ge_un1 = check(assert_type(ts >= un_idx, np_1darray_bool), np_1darray_bool)
    assert (lt_un1 != ge_un1).all()
    lt_un2 = check(assert_type(un_idx < ts, np_1darray_bool), np_1darray_bool)
    ge_un2 = check(assert_type(un_idx >= ts, np_1darray_bool), np_1darray_bool)
    assert (lt_un2 != ge_un2).all()

    # ==, !=
    eq_dt1 = check(assert_type(ts == dt_idx, np_1darray_bool), np_1darray_bool)
    ne_dt1 = check(assert_type(ts != dt_idx, np_1darray_bool), np_1darray_bool)
    assert (eq_dt1 != ne_dt1).all()
    # there is a mypy bug where ts.__eq__(Index) gets revealed as Any and not np_1darray
    eq_un1 = check(assert_type(ts == un_idx, np_1darray_bool), np_1darray_bool)  # type: ignore[assert-type]
    ne_un1 = check(assert_type(ts != un_idx, np_1darray_bool), np_1darray_bool)  # type: ignore[assert-type]
    assert (eq_un1 != ne_un1).all()

    # ==, != (ts on the rhs, use == and != of lhs)
    eq_rhs_dt1 = check(assert_type(dt_idx == ts, np_1darray_bool), np_1darray_bool)
    ne_rhs_dt1 = check(assert_type(dt_idx != ts, np_1darray_bool), np_1darray_bool)
    assert (eq_rhs_dt1 != ne_rhs_dt1).all()
    eq_rhs_un1 = check(assert_type(un_idx == ts, np_1darray_bool), np_1darray_bool)
    ne_rhs_un1 = check(assert_type(un_idx != ts, np_1darray_bool), np_1darray_bool)
    assert (eq_rhs_un1 != ne_rhs_un1).all()


def test_timestamp_cmp_array() -> None:
    ts = pd.Timestamp(year=2000, month=3, day=24, hour=12, minute=27)
    arr_1d = pd.to_datetime([1, 2, 3, 4]).to_numpy()
    arr_2d = arr_1d.reshape(2, 2)
    arr_nd = arr_1d.reshape([4])

    # >, <=
    gt_nd1 = check(assert_type(ts > arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    le_nd1 = check(assert_type(ts <= arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (gt_nd1 != le_nd1).all()
    gt_nd2 = check(assert_type(arr_nd > ts, np_ndarray_bool), np_ndarray_bool, np.bool)
    le_nd2 = check(assert_type(arr_nd <= ts, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (gt_nd2 != le_nd2).all()
    gt_2d1 = check(assert_type(ts > arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    le_2d1 = check(assert_type(ts <= arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (gt_2d1 != le_2d1).all()
    gt_2d2 = check(assert_type(arr_2d > ts, np_2darray[np.bool]), np_2darray[np.bool])
    le_2d2 = check(assert_type(arr_2d <= ts, np_2darray[np.bool]), np_2darray[np.bool])
    assert (gt_2d2 != le_2d2).all()
    gt_1d1 = check(assert_type(ts > arr_1d, np_1darray_bool), np_1darray_bool)
    le_1d1 = check(assert_type(ts <= arr_1d, np_1darray_bool), np_1darray_bool)
    assert (gt_1d1 != le_1d1).all()
    gt_1d2 = check(assert_type(arr_1d > ts, np_1darray_bool), np_1darray_bool)
    le_1d2 = check(assert_type(arr_1d <= ts, np_1darray_bool), np_1darray_bool)
    assert (gt_1d2 != le_1d2).all()

    # <, >=
    lt_nd1 = check(assert_type(ts < arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    ge_nd1 = check(assert_type(ts >= arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (lt_nd1 != ge_nd1).all()
    lt_nd2 = check(assert_type(arr_nd < ts, np_ndarray_bool), np_ndarray_bool, np.bool)
    ge_nd2 = check(assert_type(arr_nd >= ts, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (lt_nd2 != ge_nd2).all()
    lt_2d1 = check(assert_type(ts < arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    ge_2d1 = check(assert_type(ts >= arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (lt_2d1 != ge_2d1).all()
    lt_2d2 = check(assert_type(arr_2d < ts, np_2darray[np.bool]), np_2darray[np.bool])
    ge_2d2 = check(assert_type(arr_2d >= ts, np_2darray[np.bool]), np_2darray[np.bool])
    assert (lt_2d2 != ge_2d2).all()
    lt_1d1 = check(assert_type(ts < arr_1d, np_1darray_bool), np_1darray_bool)
    ge_1d1 = check(assert_type(ts >= arr_1d, np_1darray_bool), np_1darray_bool)
    assert (lt_1d1 != ge_1d1).all()
    lt_1d2 = check(assert_type(arr_1d < ts, np_1darray_bool), np_1darray_bool)
    ge_1d2 = check(assert_type(arr_1d >= ts, np_1darray_bool), np_1darray_bool)
    assert (lt_1d2 != ge_1d2).all()

    # ==, !=
    eq_nd1 = check(assert_type(ts == arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    ne_nd1 = check(assert_type(ts != arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (eq_nd1 != ne_nd1).all()
    eq_2d1 = check(assert_type(ts == arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    ne_2d1 = check(assert_type(ts != arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (eq_2d1 != ne_2d1).all()
    eq_1d1 = check(assert_type(ts == arr_1d, np_1darray_bool), np_1darray_bool)
    ne_1d1 = check(assert_type(ts != arr_1d, np_1darray_bool), np_1darray_bool)
    assert (eq_1d1 != ne_1d1).all()

    # ==, != (td on the rhs, use == and != of lhs)
    eq_rhs_nd1 = check(assert_type(arr_nd == ts, Any), np_ndarray_bool)
    ne_rhs_nd1 = check(assert_type(arr_nd != ts, Any), np_ndarray_bool)
    assert (eq_rhs_nd1 != ne_rhs_nd1).all()
    eq_rhs_2d1 = check(assert_type(arr_2d == ts, Any), np_2darray[np.bool])
    ne_rhs_2d1 = check(assert_type(arr_2d != ts, Any), np_2darray[np.bool])
    assert (eq_rhs_2d1 != ne_rhs_2d1).all()
    eq_rhs_1d1 = check(assert_type(arr_1d == ts, Any), np_1darray_bool)
    ne_rhs_1d1 = check(assert_type(arr_1d != ts, Any), np_1darray_bool)
    assert (eq_rhs_1d1 != ne_rhs_1d1).all()


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
    check(assert_type(ts.tz_localize(1, True), pd.Timestamp), pd.Timestamp)
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
        assert_type(
            ts.tz_localize("US/Pacific", nonexistent="NaT"), pd.Timestamp | NaTType
        ),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Timestamp(2025, 3, 9, 2, 30, 0).tz_localize(
                "US/Eastern", nonexistent="NaT"
            ),
            pd.Timestamp | NaTType,
        ),
        NaTType,
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

    check(assert_type(ts2.round("1s"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1s", ambiguous="raise"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1s", ambiguous=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1s", ambiguous=False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.round("1s", ambiguous="NaT"), pd.Timestamp), pd.Timestamp)

    with pytest_warns_bounded(
        FutureWarning,
        "'H' is deprecated ",
        lower="2.1.99",
        upper="2.3.99",
        upper_exception=ValueError,
    ):
        check(
            assert_type(ts2.round("2H", nonexistent="shift_forward"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(ts2.round("2H", nonexistent="shift_backward"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(ts2.round("2H", nonexistent="NaT"), pd.Timestamp), pd.Timestamp
        )
        check(
            assert_type(ts2.round("2H", nonexistent="raise"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(
                ts2.round("2H", nonexistent=pd.Timedelta(24, "h")), pd.Timestamp
            ),
            pd.Timestamp,
        )
        check(
            assert_type(
                ts2.round("2H", nonexistent=dt.timedelta(hours=24)), pd.Timestamp
            ),
            pd.Timestamp,
        )

    check(assert_type(ts2.ceil("1s"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1s", ambiguous="raise"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1s", ambiguous=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1s", ambiguous=False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.ceil("1s", ambiguous="NaT"), pd.Timestamp), pd.Timestamp)

    with pytest_warns_bounded(
        FutureWarning,
        "'H' is deprecated",
        lower="2.1.99",
        upper="2.3.99",
        upper_exception=ValueError,
    ):
        check(
            assert_type(ts2.ceil("2H", nonexistent="shift_forward"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(ts2.ceil("2H", nonexistent="shift_backward"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(ts2.ceil("2H", nonexistent="NaT"), pd.Timestamp), pd.Timestamp
        )
        check(
            assert_type(ts2.ceil("2H", nonexistent="raise"), pd.Timestamp), pd.Timestamp
        )
        check(
            assert_type(
                ts2.ceil("2H", nonexistent=pd.Timedelta(24, "h")), pd.Timestamp
            ),
            pd.Timestamp,
        )
        check(
            assert_type(
                ts2.ceil("2H", nonexistent=dt.timedelta(hours=24)), pd.Timestamp
            ),
            pd.Timestamp,
        )

    check(assert_type(ts2.floor("1s"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1s", ambiguous="raise"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1s", ambiguous=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1s", ambiguous=False), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.floor("1s", ambiguous="NaT"), pd.Timestamp), pd.Timestamp)

    with pytest_warns_bounded(
        FutureWarning,
        "'H' is deprecated",
        lower="2.1.99",
        upper="2.3.99",
        upper_exception=ValueError,
    ):
        check(
            assert_type(ts2.floor("2H", nonexistent="shift_forward"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(ts2.floor("2H", nonexistent="shift_backward"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(ts2.floor("2H", nonexistent="NaT"), pd.Timestamp), pd.Timestamp
        )
        check(
            assert_type(ts2.floor("2H", nonexistent="raise"), pd.Timestamp),
            pd.Timestamp,
        )
        check(
            assert_type(
                ts2.floor("2H", nonexistent=pd.Timedelta(24, "h")), pd.Timestamp
            ),
            pd.Timestamp,
        )
        check(
            assert_type(
                ts2.floor("2H", nonexistent=dt.timedelta(hours=24)), pd.Timestamp
            ),
            pd.Timestamp,
        )

    check(assert_type(ts2.as_unit("s"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.as_unit("ms"), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.as_unit("us", round_ok=True), pd.Timestamp), pd.Timestamp)
    check(assert_type(ts2.as_unit("ns", round_ok=False), pd.Timestamp), pd.Timestamp)


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
    check(assert_type(s, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)
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
    check(
        assert_type(
            ts.combine(dt.date(2000, 1, 1), dt.time(12, 21, 21, 12)), pd.Timestamp
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
    check(
        assert_type(
            pd.Period(value=datetime.datetime(2012, 1, 1), freq="D"), pd.Period
        ),
        pd.Period,
    )
    check(
        assert_type(pd.Period(value=datetime.date(2012, 1, 1), freq="D"), pd.Period),
        pd.Period,
    )
    check(
        assert_type(pd.Period(value=pd.Timestamp(2012, 1, 1), freq="D"), pd.Period),
        pd.Period,
    )


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
    as_int: int = 1
    as_period_index = pd.period_range("2012-1-1", periods=10, freq="D")
    check(assert_type(as_period_index, pd.PeriodIndex), pd.PeriodIndex)
    as_period = pd.Period("2012-1-1", freq="D")
    scale = 24 * 60 * 60 * 10**9
    as_td_series = pd.Series(pd.timedelta_range(scale, scale, freq="D"))
    check(assert_type(as_td_series, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)
    as_period_series = pd.Series(as_period_index)
    check(assert_type(as_period_series, "pd.Series[pd.Period]"), pd.Series, pd.Period)
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

    check(assert_type(p + as_td_series, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(p + as_timedelta_idx, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p + as_nat, NaTType), NaTType)
    offset_series = as_period_series - as_period_series
    check(assert_type(offset_series, "pd.Series[BaseOffset]"), pd.Series)
    check(assert_type(p + offset_series, "pd.Series[pd.Period]"), pd.Series, pd.Period)
    check(assert_type(p - as_pd_td, pd.Period), pd.Period)
    check(assert_type(p - as_dt_td, pd.Period), pd.Period)
    check(assert_type(p - as_np_td, pd.Period), pd.Period)
    check(assert_type(p - as_np_i64, pd.Period), pd.Period)
    check(assert_type(p - as_int, pd.Period), pd.Period)
    check(assert_type(offset_index, pd.Index), pd.Index)
    check(assert_type(p - as_period, BaseOffset), Day)
    check(assert_type(p - as_td_series, "pd.Series[pd.Period]"), pd.Series, pd.Period)
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

    check(assert_type(as_td_series + p, "pd.Series[pd.Period]"), pd.Series, pd.Period)

    check(assert_type(as_timedelta_idx + p, pd.PeriodIndex), pd.PeriodIndex)

    check(assert_type(as_nat + p, NaTType), NaTType)
    check(assert_type(p.__radd__(as_nat), NaTType), NaTType)

    check(assert_type(p.freq + p, pd.Period), pd.Period)
    check(assert_type(p.__radd__(p.freq), pd.Period), pd.Period)

    check(assert_type(as_period_index - p, pd.Index), pd.Index)


def test_period_cmp_scalar() -> None:
    p = pd.Period("2012-1-1", freq="D")
    p2 = pd.Period("2012-1-2", freq="D")

    # >, <=
    gt1 = check(assert_type(p > p2, bool), bool)
    le1 = check(assert_type(p <= p2, bool), bool)
    assert gt1 != le1

    # <, >=
    lt1 = check(assert_type(p < p2, bool), bool)
    ge1 = check(assert_type(p >= p2, bool), bool)
    assert lt1 != ge1

    # ==, !=
    eq1 = check(assert_type(p == p2, bool), bool)
    ne1 = check(assert_type(p != p2, bool), bool)
    assert eq1 != ne1
    eq2 = check(assert_type(p == 1, Literal[False]), bool)
    ne2 = check(assert_type(p != 1, Literal[True]), bool)
    assert eq2 != ne2


def test_period_cmp_series() -> None:
    p = pd.Period("2012-1-1", freq="D")
    p_ser = pd.Series(pd.period_range("2012-1-1", periods=10, freq="D"))

    # >, <=
    gt1 = check(assert_type(p > p_ser, "pd.Series[bool]"), pd.Series, np.bool)
    le1 = check(assert_type(p <= p_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (gt1 != le1).all()
    gt2 = check(assert_type(p_ser > p, "pd.Series[bool]"), pd.Series, np.bool)
    le2 = check(assert_type(p_ser <= p, "pd.Series[bool]"), pd.Series, np.bool)
    assert (gt2 != le2).all()

    # <, >=
    lt1 = check(assert_type(p < p_ser, "pd.Series[bool]"), pd.Series, np.bool)
    ge1 = check(assert_type(p >= p_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (lt1 != ge1).all()
    lt2 = check(assert_type(p_ser < p, "pd.Series[bool]"), pd.Series, np.bool)
    ge2 = check(assert_type(p_ser >= p, "pd.Series[bool]"), pd.Series, np.bool)
    assert (lt2 != ge2).all()

    # ==, !=
    eq1 = check(assert_type(p == p_ser, "pd.Series[bool]"), pd.Series, np.bool)
    ne1 = check(assert_type(p != p_ser, "pd.Series[bool]"), pd.Series, np.bool)
    assert (eq1 != ne1).all()

    # ==, != (p on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(p_ser == p, "pd.Series[bool]"), pd.Series, np.bool)
    ne_rhs1 = check(assert_type(p_ser != p, "pd.Series[bool]"), pd.Series, np.bool)
    assert (eq_rhs1 != ne_rhs1).all()


def test_period_cmp_index() -> None:
    p = pd.Period("2012-1-1", freq="D")
    p_idx = pd.period_range("2012-1-1", periods=10, freq="D")

    # >, <=
    gt1 = check(assert_type(p > p_idx, np_1darray_bool), np_1darray_bool)
    le1 = check(assert_type(p <= p_idx, np_1darray_bool), np_1darray_bool)
    assert (gt1 != le1).all()
    gt2 = check(assert_type(p_idx > p, np_1darray_bool), np_1darray_bool)
    le2 = check(assert_type(p_idx <= p, np_1darray_bool), np_1darray_bool)
    assert (gt2 != le2).all()

    # <, >=
    lt1 = check(assert_type(p < p_idx, np_1darray_bool), np_1darray_bool)
    ge1 = check(assert_type(p >= p_idx, np_1darray_bool), np_1darray_bool)
    assert (lt1 != ge1).all()
    lt2 = check(assert_type(p_idx < p, np_1darray_bool), np_1darray_bool)
    ge2 = check(assert_type(p_idx >= p, np_1darray_bool), np_1darray_bool)
    assert (lt2 != ge2).all()

    # ==, !=
    eq1 = check(assert_type(p == p_idx, np_1darray_bool), np_1darray_bool)
    ne1 = check(assert_type(p != p_idx, np_1darray_bool), np_1darray_bool)
    assert (eq1 != ne1).all()

    # ==, != (p on the rhs, use == and != of lhs)
    eq_rhs1 = check(assert_type(p_idx == p, np_1darray_bool), np_1darray_bool)
    ne_rhs1 = check(assert_type(p_idx != p, np_1darray_bool), np_1darray_bool)
    assert (eq_rhs1 != ne_rhs1).all()


def test_period_cmp_array() -> None:
    p = pd.Period("2012-1-1", freq="D")
    arr_1d = pd.period_range("2012-1-1", periods=4, freq="D").to_numpy()
    arr_2d = arr_1d.reshape(2, 2)
    arr_nd = arr_1d.reshape([4])

    # >, <=
    gt_nd1 = check(assert_type(p > arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    le_nd1 = check(assert_type(p <= arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (gt_nd1 != le_nd1).all()
    gt_2d1 = check(assert_type(p > arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    le_2d1 = check(assert_type(p <= arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (gt_2d1 != le_2d1).all()
    gt_1d1 = check(assert_type(p > arr_1d, np_1darray_bool), np_1darray_bool)
    le_1d1 = check(assert_type(p <= arr_1d, np_1darray_bool), np_1darray_bool)
    assert (gt_1d1 != le_1d1).all()
    # p on the rhs, type depends on np.ndarray > and <= methods
    gt_nd2 = check(assert_type(arr_nd > p, np_ndarray_bool), np_ndarray_bool, np.bool)
    le_nd2 = check(assert_type(arr_nd <= p, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (gt_nd2 != le_nd2).all()
    gt_2d2 = check(assert_type(arr_2d > p, np_ndarray_bool), np_2darray[np.bool])
    le_2d2 = check(assert_type(arr_2d <= p, np_ndarray_bool), np_2darray[np.bool])
    assert (gt_2d2 != le_2d2).all()
    gt_1d2 = check(assert_type(arr_1d > p, np_ndarray_bool), np_1darray_bool)
    le_1d2 = check(assert_type(arr_1d <= p, np_ndarray_bool), np_1darray_bool)
    assert (gt_1d2 != le_1d2).all()

    # <, >=
    lt_nd1 = check(assert_type(p < arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    ge_nd1 = check(assert_type(p >= arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (lt_nd1 != ge_nd1).all()
    lt_2d1 = check(assert_type(p < arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    ge_2d1 = check(assert_type(p >= arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (lt_2d1 != ge_2d1).all()
    lt_1d1 = check(assert_type(p < arr_1d, np_1darray_bool), np_1darray_bool)
    ge_1d1 = check(assert_type(p >= arr_1d, np_1darray_bool), np_1darray_bool)
    assert (lt_1d1 != ge_1d1).all()
    # p on the rhs, type depends on np.ndarray < and >= methods
    lt_nd2 = check(assert_type(arr_nd < p, np_ndarray_bool), np_ndarray_bool, np.bool)
    ge_nd2 = check(assert_type(arr_nd >= p, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (lt_nd2 != ge_nd2).all()
    lt_2d2 = check(assert_type(arr_2d < p, np_ndarray_bool), np_2darray[np.bool])
    ge_2d2 = check(assert_type(arr_2d >= p, np_ndarray_bool), np_2darray[np.bool])
    assert (lt_2d2 != ge_2d2).all()
    lt_1d2 = check(assert_type(arr_1d < p, np_ndarray_bool), np_1darray_bool)
    ge_1d2 = check(assert_type(arr_1d >= p, np_ndarray_bool), np_1darray_bool)
    assert (lt_1d2 != ge_1d2).all()

    # ==, !=
    eq_nd1 = check(assert_type(p == arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    ne_nd1 = check(assert_type(p != arr_nd, np_ndarray_bool), np_ndarray_bool, np.bool)
    assert (eq_nd1 != ne_nd1).all()
    eq_2d1 = check(assert_type(p == arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    ne_2d1 = check(assert_type(p != arr_2d, np_2darray[np.bool]), np_2darray[np.bool])
    assert (eq_2d1 != ne_2d1).all()
    eq_1d1 = check(assert_type(p == arr_1d, np_1darray_bool), np_1darray_bool)
    ne_1d1 = check(assert_type(p != arr_1d, np_1darray_bool), np_1darray_bool)
    assert (eq_1d1 != ne_1d1).all()

    # ==, != (td on the rhs, use == and != of lhs)
    eq_rhs_nd1 = check(assert_type(arr_nd == p, Any), np_ndarray_bool)
    ne_rhs_nd1 = check(assert_type(arr_nd != p, Any), np_ndarray_bool)
    assert (eq_rhs_nd1 != ne_rhs_nd1).all()
    eq_rhs_2d1 = check(assert_type(arr_2d == p, Any), np_2darray[np.bool])
    ne_rhs_2d1 = check(assert_type(arr_2d != p, Any), np_2darray[np.bool])
    assert (eq_rhs_2d1 != ne_rhs_2d1).all()
    eq_rhs_1d1 = check(assert_type(arr_1d == p, Any), np_1darray_bool)
    ne_rhs_1d1 = check(assert_type(arr_1d != p, Any), np_1darray_bool)
    assert (eq_rhs_1d1 != ne_rhs_1d1).all()


def test_period_methods() -> None:
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


def test_nattype_hashable() -> None:
    # GH 827
    check(assert_type(pd.NaT.__hash__(), int), int)
