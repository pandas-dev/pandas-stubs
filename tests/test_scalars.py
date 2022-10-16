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
