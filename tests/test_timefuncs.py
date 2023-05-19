# flake8: noqa: F841
from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

import numpy as np
from numpy import typing as npt
import pandas as pd
import pytz
from typing_extensions import (
    assert_never,
    assert_type,
)

from pandas._libs import NaTType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.offsets import DateOffset

if TYPE_CHECKING:
    from pandas._typing import FulldatetimeDict
else:
    FulldatetimeDict = Any
from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import (
    BusinessDay,
    BusinessHour,
    CustomBusinessDay,
    CustomBusinessHour,
    Day,
)

if TYPE_CHECKING:
    from pandas.core.series import (
        PeriodSeries,
        TimedeltaSeries,
        TimestampSeries,
    )

if TYPE_CHECKING:
    from pandas.core.indexes.base import _IntIndexType

else:
    from pandas import Index as _IntIndexType

# Separately define here so pytest works
np_ndarray_bool = npt.NDArray[np.bool_]


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

    tsr: pd.Timedelta = ts - ts2
    tsr2: pd.Timestamp = ts + delta
    tsr3: pd.Timestamp = ts - delta
    tsr4: pd.Timedelta = ts - dt.datetime(2021, 1, 3)


def test_types_comparison() -> None:
    ts: pd.Timestamp = pd.to_datetime("2021-03-01")
    ts2: pd.Timestamp = pd.to_datetime("2021-01-01")

    tsr: bool = ts < ts2
    tsr2: bool = ts > ts2


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
    check(assert_type(s, "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(ts2 <= s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts2 >= s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts2 < s, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(ts2 > s, "pd.Series[bool]"), pd.Series, np.bool_)


def test_types_pydatetime() -> None:
    ts: pd.Timestamp = pd.Timestamp("2021-03-01T12")

    datet: dt.datetime = ts.to_pydatetime()
    datet2: dt.datetime = ts.to_pydatetime(False)
    datet3: dt.datetime = ts.to_pydatetime(warn=True)


def test_to_timedelta() -> None:
    td: pd.Timedelta = pd.to_timedelta(3, "days")
    tds: pd.TimedeltaIndex = pd.to_timedelta([2, 3], "minutes")


def test_timedelta_arithmetic() -> None:
    td1: pd.Timedelta = pd.to_timedelta(3, "days")
    td2: pd.Timedelta = pd.to_timedelta(4, "hours")
    td3: pd.Timedelta = td1 + td2
    td4: pd.Timedelta = td1 - td2
    td5: pd.Timedelta = td1 * 4.3
    td6: pd.Timedelta = td3 / 10.2


def test_timedelta_series_arithmetic() -> None:
    tds1: pd.TimedeltaIndex = pd.to_timedelta([2, 3], "minutes")
    td1: pd.Timedelta = pd.Timedelta("2 days")
    r1: pd.TimedeltaIndex = tds1 + td1
    r2: pd.TimedeltaIndex = tds1 - td1
    r3: pd.TimedeltaIndex = tds1 * 4.3
    r4: pd.TimedeltaIndex = tds1 / 10.2


def test_timestamp_timedelta_series_arithmetic() -> None:
    ts1 = pd.to_datetime(pd.Series(["2022-03-05", "2022-03-06"]))
    assert isinstance(ts1.iloc[0], pd.Timestamp)
    td1 = pd.to_timedelta([2, 3], "seconds")
    ts2 = pd.to_datetime(pd.Series(["2022-03-08", "2022-03-10"]))
    r1 = ts1 - ts2
    check(assert_type(r1, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    r2 = r1 / td1
    check(assert_type(r2, "pd.Series[float]"), pd.Series, float)
    r3 = r1 - td1
    check(assert_type(r3, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    r4 = pd.Timedelta(5, "days") / r1
    check(assert_type(r4, "pd.Series[float]"), pd.Series, float)
    sb = pd.Series([1, 2]) == pd.Series([1, 3])
    check(assert_type(sb, "pd.Series[bool]"), pd.Series, np.bool_)
    r5 = sb * r1
    check(assert_type(r5, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    r6 = r1 * 4
    check(assert_type(r6, "TimedeltaSeries"), pd.Series, pd.Timedelta)

    tsp1 = pd.Timestamp("2022-03-05")
    dt1 = dt.datetime(2022, 9, 1, 12, 5, 30)
    r7 = ts1 - tsp1
    check(assert_type(r7, "TimedeltaSeries"), pd.Series, pd.Timedelta)
    r8 = ts1 - dt1
    check(assert_type(r8, "TimedeltaSeries"), pd.Series, pd.Timedelta)


def test_timestamp_dateoffset_arithmetic() -> None:
    ts = pd.Timestamp("2022-03-18")
    do = pd.DateOffset(days=366)
    r1: pd.Timestamp = ts + do


def test_datetimeindex_plus_timedelta() -> None:
    tscheck = pd.Series([pd.Timestamp("2022-03-05"), pd.Timestamp("2022-03-06")])
    dti = pd.to_datetime(["2022-03-08", "2022-03-15"])
    td_s = pd.to_timedelta(pd.Series([10, 20]), "minutes")
    dti_td_s = dti + td_s
    check(
        assert_type(dti_td_s, "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    td_dti_s = td_s + dti
    check(
        assert_type(td_dti_s, "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    tdi = pd.to_timedelta([10, 20], "minutes")
    dti_tdi_dti = dti + tdi
    check(assert_type(dti_tdi_dti, "pd.DatetimeIndex"), pd.DatetimeIndex)
    tdi_dti_dti = tdi + dti
    check(assert_type(tdi_dti_dti, "pd.DatetimeIndex"), pd.DatetimeIndex)
    dti_td_dti = dti + pd.Timedelta(10, "minutes")
    check(assert_type(dti_td_dti, "pd.DatetimeIndex"), pd.DatetimeIndex)
    ts_tdi_dti = pd.Timestamp("2022-03-05") + tdi
    check(assert_type(ts_tdi_dti, pd.DatetimeIndex), pd.DatetimeIndex)


def test_datetimeindex_minus_timedelta() -> None:
    # GH 280
    tscheck = pd.Series([pd.Timestamp("2022-03-05"), pd.Timestamp("2022-03-06")])
    dti = pd.to_datetime(["2022-03-08", "2022-03-15"])
    td_s = pd.to_timedelta(pd.Series([10, 20]), "minutes")
    dti_td_s = dti - td_s
    check(
        assert_type(dti_td_s, "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    tdi = pd.to_timedelta([10, 20], "minutes")
    dti_tdi_dti = dti - tdi
    check(assert_type(dti_tdi_dti, "pd.DatetimeIndex"), pd.DatetimeIndex)
    dti_td_dti = dti - pd.Timedelta(10, "minutes")
    check(assert_type(dti_td_dti, "pd.DatetimeIndex"), pd.DatetimeIndex)
    dti_ts_tdi = dti - pd.Timestamp("2022-03-05")
    check(assert_type(dti_ts_tdi, pd.TimedeltaIndex), pd.TimedeltaIndex)


def test_timestamp_plus_timedelta_series() -> None:
    tscheck = pd.Series([pd.Timestamp("2022-03-05"), pd.Timestamp("2022-03-06")])
    ts = pd.Timestamp("2022-03-05")
    td = pd.to_timedelta(pd.Series([10, 20]), "minutes")
    r3 = td + ts
    check(assert_type(r3, "TimestampSeries"), pd.Series, pd.Timestamp)
    r4 = ts + td
    check(assert_type(r4, "TimestampSeries"), pd.Series, pd.Timestamp)


def test_timedelta_series_mult() -> None:
    df = pd.DataFrame({"x": [1, 3, 5], "y": [2, 2, 6]})
    std = (df["x"] < df["y"]) * pd.Timedelta(10, "minutes")
    check(
        assert_type(std, "TimedeltaSeries"),
        pd.Series,
        pd.Timedelta,
    )


def test_timedelta_series_sum() -> None:
    s = pd.Series(pd.to_datetime(["04/05/2022 11:00", "04/03/2022 10:00"])) - pd.Series(
        pd.to_datetime(["04/05/2022 08:00", "04/03/2022 09:00"])
    )
    ssum = s.sum()
    ires: int = ssum.days

    sf = pd.Series([1.0, 2.2, 3.3])
    sfsum: float = sf.sum()


def test_iso_calendar() -> None:
    # GH 31
    dates = pd.date_range(start="2012-01-01", end="2019-12-31", freq="W-MON")
    dates.isocalendar()


def test_fail_on_adding_two_timestamps() -> None:
    s1 = pd.Series(pd.to_datetime(["2022-05-01", "2022-06-01"]))
    s2 = pd.Series(pd.to_datetime(["2022-05-15", "2022-06-15"]))
    if TYPE_CHECKING_INVALID_USAGE:
        ssum: pd.Series = s1 + s2  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        ts = pd.Timestamp("2022-06-30")
        tsum: pd.Series = s1 + ts  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]


def test_dtindex_tzinfo() -> None:
    # GH 71
    dti = pd.date_range("2000-1-1", periods=10)
    assert assert_type(dti.tzinfo, Optional[dt.tzinfo]) is None


def test_todatetime_fromnumpy() -> None:
    # GH 72
    t1 = np.datetime64("2022-07-04 02:30")
    check(assert_type(pd.to_datetime(t1), pd.Timestamp), pd.Timestamp)


def test_comparisons_datetimeindex() -> None:
    # GH 74
    dti = pd.date_range("2000-01-01", "2000-01-10")
    ts = pd.Timestamp("2000-01-05")
    check(assert_type((dti < ts), np_ndarray_bool), np.ndarray)
    check(assert_type((dti > ts), np_ndarray_bool), np.ndarray)
    check(assert_type((dti >= ts), np_ndarray_bool), np.ndarray)
    check(assert_type((dti <= ts), np_ndarray_bool), np.ndarray)
    check(assert_type((dti == ts), np_ndarray_bool), np.ndarray)
    check(assert_type((dti != ts), np_ndarray_bool), np.ndarray)


def test_to_datetime_nat() -> None:
    # GH 88
    check(
        assert_type(pd.to_datetime("2021-03-01", errors="ignore"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(pd.to_datetime("2021-03-01", errors="raise"), pd.Timestamp),
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.to_datetime("2021-03-01", errors="coerce"),
            "Union[pd.Timestamp, NaTType]",
        ),
        pd.Timestamp,
    )

    check(
        assert_type(
            pd.to_datetime("not a date", errors="coerce"),
            "Union[pd.Timestamp, NaTType]",
        ),
        NaTType,
    )


def test_series_dt_accessors() -> None:
    # GH 131
    i0 = pd.date_range(start="2022-06-01", periods=10)
    check(assert_type(i0, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(i0.to_series(), "TimestampSeries"), pd.Series, pd.Timestamp)

    s0 = pd.Series(i0)

    check(assert_type(s0.dt.date, "pd.Series[dt.date]"), pd.Series, dt.date)
    check(assert_type(s0.dt.time, "pd.Series[dt.time]"), pd.Series, dt.time)
    check(assert_type(s0.dt.timetz, "pd.Series[dt.time]"), pd.Series, dt.time)
    check(assert_type(s0.dt.year, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.month, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.day, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.hour, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.minute, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.second, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.microsecond, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.nanosecond, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.dayofweek, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.day_of_week, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.weekday, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.dayofyear, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.day_of_year, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.quarter, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.is_month_start, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.is_month_end, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.is_quarter_start, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.is_quarter_end, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.is_year_start, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.is_year_end, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.is_leap_year, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(s0.dt.daysinmonth, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s0.dt.days_in_month, "pd.Series[int]"), pd.Series, np.integer)
    assert assert_type(s0.dt.tz, Optional[dt.tzinfo]) is None
    check(assert_type(s0.dt.freq, Optional[str]), str)
    check(assert_type(s0.dt.isocalendar(), pd.DataFrame), pd.DataFrame)
    check(assert_type(s0.dt.to_period("D"), "PeriodSeries"), pd.Series, pd.Period)

    check(assert_type(s0.dt.to_pydatetime(), np.ndarray), np.ndarray, dt.datetime)
    s0_local = s0.dt.tz_localize("UTC")
    check(
        assert_type(s0_local, "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(s0.dt.tz_localize(None), "TimestampSeries"), pd.Series, pd.Timestamp
    )
    check(
        assert_type(s0.dt.tz_localize(pytz.UTC), "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(s0.dt.tz_localize(pytz.timezone("US/Eastern")), "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(s0_local.dt.tz_convert("EST"), "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(s0_local.dt.tz_convert(None), "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(s0_local.dt.tz_convert(pytz.UTC), "TimestampSeries"),
        pd.Series,
        pd.Timestamp,
    )
    check(
        assert_type(
            s0_local.dt.tz_convert(pytz.timezone("US/Eastern")), "TimestampSeries"
        ),
        pd.Series,
        pd.Timestamp,
    )
    check(assert_type(s0.dt.tz, Optional[dt.tzinfo]), type(None))
    check(assert_type(s0_local.dt.tz, Optional[dt.tzinfo]), dt.tzinfo)
    check(assert_type(s0.dt.normalize(), "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(s0.dt.strftime("%Y"), "pd.Series[str]"), pd.Series, str)
    check(assert_type(s0.dt.round("D"), "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(s0.dt.floor("D"), "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(s0.dt.ceil("D"), "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(s0.dt.month_name(), "pd.Series[str]"), pd.Series, str)
    check(assert_type(s0.dt.day_name(), "pd.Series[str]"), pd.Series, str)

    i1 = pd.period_range(start="2022-06-01", periods=10)

    check(assert_type(i1, pd.PeriodIndex), pd.PeriodIndex)

    check(assert_type(i1.to_series(), pd.Series), pd.Series, pd.Period)

    s1 = pd.Series(i1)

    check(assert_type(s1.dt.qyear, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s1.dt.start_time, "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type(s1.dt.end_time, "TimestampSeries"), pd.Series, pd.Timestamp)

    i2 = pd.timedelta_range(start="1 day", periods=10)
    check(assert_type(i2, pd.TimedeltaIndex), pd.TimedeltaIndex)

    check(assert_type(i2.to_series(), "TimedeltaSeries"), pd.Series, pd.Timedelta)

    s2 = pd.Series(i2)

    check(assert_type(s2.dt.days, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2.dt.seconds, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2.dt.microseconds, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2.dt.nanoseconds, "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s2.dt.components, pd.DataFrame), pd.DataFrame)
    check(assert_type(s2.dt.to_pytimedelta(), np.ndarray), np.ndarray)
    check(assert_type(s2.dt.total_seconds(), "pd.Series[float]"), pd.Series, float)


def test_datetimeindex_accessors() -> None:
    # GH 194
    x = pd.DatetimeIndex(["2022-08-14", "2022-08-20"])
    check(assert_type(x.month, "_IntIndexType"), _IntIndexType)

    i0 = pd.date_range(start="2022-06-01", periods=10)
    check(assert_type(i0, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)

    check(assert_type(i0.date, np.ndarray), np.ndarray, dt.date)
    check(assert_type(i0.time, np.ndarray), np.ndarray, dt.time)
    check(assert_type(i0.timetz, np.ndarray), np.ndarray, dt.time)
    check(assert_type(i0.year, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.month, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.day, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.hour, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.minute, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.second, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.microsecond, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.nanosecond, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.dayofweek, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.day_of_week, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.weekday, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.dayofyear, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.day_of_year, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.quarter, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.is_month_start, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.is_month_end, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.is_quarter_start, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.is_quarter_end, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.is_year_start, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.is_year_end, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.is_leap_year, npt.NDArray[np.bool_]), np.ndarray, np.bool_)
    check(assert_type(i0.daysinmonth, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.days_in_month, "_IntIndexType"), _IntIndexType, np.int32)
    check(assert_type(i0.tz, Optional[dt.tzinfo]), type(None))
    check(assert_type(i0.freq, Optional[BaseOffset]), BaseOffset)
    check(assert_type(i0.isocalendar(), pd.DataFrame), pd.DataFrame)
    check(assert_type(i0.to_period("D"), pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(
        assert_type(i0.to_pydatetime(), npt.NDArray[np.object_]),
        np.ndarray,
        dt.datetime,
    )
    ilocal = i0.tz_localize("UTC")
    check(assert_type(ilocal, pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(i0.tz_localize(pytz.UTC), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(i0.tz_localize(pytz.timezone("US/Central")), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(assert_type(ilocal.tz_convert("EST"), pd.DatetimeIndex), pd.DatetimeIndex)
    check(assert_type(ilocal.tz_convert(pytz.UTC), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(ilocal.tz_convert(pytz.timezone("US/Pacific")), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(assert_type(ilocal.tz, Optional[dt.tzinfo]), dt.tzinfo)
    check(assert_type(i0.normalize(), pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(i0.strftime("%Y"), pd.Index), pd.Index, str)
    check(assert_type(i0.round("D"), pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(i0.floor("D"), pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(i0.ceil("D"), pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(i0.month_name(), pd.Index), pd.Index, str)
    check(assert_type(i0.day_name(), pd.Index), pd.Index, str)
    check(assert_type(i0.is_normalized, bool), bool)


def test_timedeltaindex_accessors() -> None:
    # GH 292
    i0 = pd.date_range("1/1/2021", "1/5/2021") - pd.Timestamp("1/3/2019")
    check(assert_type(i0, pd.TimedeltaIndex), pd.TimedeltaIndex)
    check(assert_type(i0.days, pd.Index), pd.Index, np.integer)
    check(assert_type(i0.seconds, pd.Index), pd.Index, np.integer)
    check(assert_type(i0.microseconds, pd.Index), pd.Index, np.integer)
    check(assert_type(i0.nanoseconds, pd.Index), pd.Index, np.integer)
    check(assert_type(i0.components, pd.DataFrame), pd.DataFrame)
    check(assert_type(i0.to_pytimedelta(), np.ndarray), np.ndarray)
    check(assert_type(i0.total_seconds(), pd.Index), pd.Index, float)
    check(
        assert_type(i0.round("D"), pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta
    )
    check(
        assert_type(i0.floor("D"), pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta
    )
    check(assert_type(i0.ceil("D"), pd.TimedeltaIndex), pd.TimedeltaIndex, pd.Timedelta)


def test_periodindex_accessors() -> None:
    # GH 395

    i0 = pd.period_range(start="2022-06-01", periods=10)
    check(assert_type(i0, pd.PeriodIndex), pd.PeriodIndex, pd.Period)

    check(assert_type(i0.year, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.month, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.day, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.hour, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.minute, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.second, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.dayofweek, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.day_of_week, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.weekday, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.dayofyear, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.day_of_year, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.quarter, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.daysinmonth, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.days_in_month, "_IntIndexType"), _IntIndexType, np.integer)
    check(assert_type(i0.freq, Optional[BaseOffset]), BaseOffset)
    check(assert_type(i0.strftime("%Y"), pd.Index), pd.Index, str)
    check(assert_type(i0.asfreq("D"), pd.PeriodIndex), pd.PeriodIndex, pd.Period)
    check(assert_type(i0.end_time, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(assert_type(i0.start_time, pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp)
    check(
        assert_type(i0.to_timestamp(), pd.DatetimeIndex), pd.DatetimeIndex, pd.Timestamp
    )
    check(assert_type(i0.freqstr, str), str)


def test_some_offsets() -> None:
    # GH 222
    check(
        assert_type(
            CustomBusinessDay(calendar=USFederalHolidayCalendar()), CustomBusinessDay
        ),
        CustomBusinessDay,
    )
    # GH 223
    check(
        assert_type(pd.date_range("1/1/2022", "2/1/2022", freq="1D"), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.date_range("1/1/2022", "2/1/2022", freq=Day()), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.bdate_range("1/1/2022", "2/1/2022", freq=BusinessDay()), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    # GH 224
    check(assert_type(dt.date.today() - Day(), dt.date), dt.date)
    # GH 235
    check(
        assert_type(
            pd.date_range("1/1/2022", "2/1/2022", freq=dt.timedelta(days=2)),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.bdate_range("1/1/2022", "2/1/2022", freq=dt.timedelta(days=2)),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.date_range(
                dt.date(2022, 1, 1), dt.date(2022, 2, 1), freq=dt.timedelta(days=2)
            ),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.bdate_range(
                dt.date(2022, 1, 1), dt.date(2022, 2, 1), freq=dt.timedelta(days=2)
            ),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.date_range("1/1/2022", "2/1/2022", freq=pd.Timedelta(days=5)),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.bdate_range("1/1/2022", "2/1/2022", freq=pd.Timedelta(days=5)),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    # GH 320
    tswm1 = pd.Timestamp("9/23/2022") + pd.offsets.WeekOfMonth(2, True)
    check(assert_type(tswm1, pd.Timestamp), pd.Timestamp)
    tswm2 = pd.Timestamp("9/23/2022") + pd.offsets.LastWeekOfMonth(2, 3)
    check(assert_type(tswm2, pd.Timestamp), pd.Timestamp)
    # GH 396
    check(
        assert_type(
            BusinessHour(start=dt.time(9, 30), end=dt.time(16, 0)), BusinessHour
        ),
        BusinessHour,
    )
    check(
        assert_type(BusinessHour(start="9:30", end="16:00"), BusinessHour), BusinessHour
    )
    check(
        assert_type(
            BusinessHour(
                start=["9:30", dt.time(11, 30)], end=[dt.time(10, 30), "13:00"]
            ),
            BusinessHour,
        ),
        BusinessHour,
    )
    check(
        assert_type(
            CustomBusinessHour(start=dt.time(9, 30), end=dt.time(16, 0)),
            CustomBusinessHour,
        ),
        CustomBusinessHour,
    )
    check(
        assert_type(CustomBusinessHour(start="9:30", end="16:00"), CustomBusinessHour),
        CustomBusinessHour,
    )
    check(
        assert_type(
            CustomBusinessHour(
                start=["9:30", dt.time(11, 30)], end=[dt.time(10, 30), "13:00"]
            ),
            CustomBusinessHour,
        ),
        CustomBusinessHour,
    )


def test_types_to_numpy() -> None:
    td_s = pd.to_timedelta(pd.Series([10, 20]), "minutes")
    check(assert_type(td_s.to_numpy(), np.ndarray), np.ndarray)
    check(assert_type(td_s.to_numpy(dtype="int", copy=True), np.ndarray), np.ndarray)
    check(assert_type(td_s.to_numpy(na_value=pd.Timedelta(0)), np.ndarray), np.ndarray)


def test_to_timdelta_units() -> None:
    check(assert_type(pd.to_timedelta(1, "W"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "w"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "D"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "d"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "days"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "day"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "hours"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "hour"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "hr"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "h"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "m"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "minute"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "min"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "minutes"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "t"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "s"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "seconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "sec"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "second"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "ms"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "milliseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "millisecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "milli"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "millis"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "l"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "us"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "microseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "microsecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "µs"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "micro"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "micros"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "u"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "ns"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "nanoseconds"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "nano"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "nanos"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "nanosecond"), pd.Timedelta), pd.Timedelta)
    check(assert_type(pd.to_timedelta(1, "n"), pd.Timedelta), pd.Timedelta)


def test_to_timedelta_scalar() -> None:
    check(
        assert_type(pd.to_timedelta(10, "ms", errors="raise"), pd.Timedelta),
        pd.Timedelta,
    )
    check(
        assert_type(pd.to_timedelta("10ms", errors="ignore"), pd.Timedelta),
        pd.Timedelta,
    )
    check(
        assert_type(
            pd.to_timedelta(dt.timedelta(milliseconds=10), errors="coerce"),
            pd.Timedelta,
        ),
        pd.Timedelta,
    )


def test_to_timedelta_series() -> None:
    s = pd.Series([10, 20, 30, 40])
    s2 = pd.Series(["10ms", "20ms", "30ms"])
    check(assert_type(pd.to_timedelta(s, "ms"), "TimedeltaSeries"), pd.Series)
    check(assert_type(pd.to_timedelta(s2), "TimedeltaSeries"), pd.Series)


def test_to_timedelta_index() -> None:
    arg0 = [1.0, 2.0, 3.0]
    arg1 = [
        dt.timedelta(milliseconds=1),
        dt.timedelta(milliseconds=2),
        dt.timedelta(milliseconds=3),
    ]
    arg2 = tuple(arg0)
    arg3 = tuple(arg1)
    arg4 = range(0, 10)
    arg5 = np.arange(10)
    arg6 = pd.Index(arg5)
    check(
        assert_type(pd.to_timedelta(arg0, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.to_timedelta(arg1, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.to_timedelta(arg2, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.to_timedelta(arg3, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.to_timedelta(arg4, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.to_timedelta(arg5, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )
    check(
        assert_type(pd.to_timedelta(arg6, "ms"), pd.TimedeltaIndex), pd.TimedeltaIndex
    )


def test_bdate_range_holidays() -> None:
    pd.bdate_range("2000-1-1", "2001-1-1", freq="C", holidays=["2000-12-15"])
    pd.bdate_range("2000-1-1", "2001-1-1", freq="C", holidays=[dt.date(2000, 12, 15)])
    pd.bdate_range(
        "2000-1-1", "2001-1-1", freq="C", holidays=[pd.Timestamp(2000, 12, 15)]
    )
    pd.bdate_range(
        "2000-1-1", "2001-1-1", freq="C", holidays=[np.datetime64("2000-12-15")]
    )
    pd.bdate_range(
        "2000-1-1", "2001-1-1", freq="C", holidays=[dt.datetime(2000, 12, 15)]
    )
    pd.bdate_range(
        "2000-1-1", "2001-1-1", freq="C", holidays=[dt.date(2000, 12, 15)], name=("a",)
    )


def test_period_range() -> None:
    check(
        assert_type(
            pd.period_range(pd.Period("2001Q1"), end=pd.Period("2010Q1")),
            pd.PeriodIndex,
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(pd.period_range("2001Q1", end=pd.Period("2010Q1")), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.period_range("2001-01-01", end="2010-01-01", freq="Q"), pd.PeriodIndex
        ),
        pd.PeriodIndex,
    )
    check(
        assert_type(pd.period_range("2001Q1", periods=100, freq="Q"), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(pd.period_range("2001Q1", periods=100, freq=Day()), pd.PeriodIndex),
        pd.PeriodIndex,
    )
    check(
        assert_type(
            pd.period_range("2001Q1", periods=100, freq=Day(), name=("A",)),
            pd.PeriodIndex,
        ),
        pd.PeriodIndex,
    )


def test_to_datetime_scalar() -> None:
    check(assert_type(pd.to_datetime("2000-01-01"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.to_datetime(1), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.to_datetime(1.5), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.to_datetime(dt.datetime(2000, 1, 1)), pd.Timestamp), pd.Timestamp
    )
    check(assert_type(pd.to_datetime(dt.date(2000, 1, 1)), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.to_datetime(np.datetime64("2000-01-01")), pd.Timestamp),
        pd.Timestamp,
    )


def test_to_datetime_scalar_extended() -> None:
    check(assert_type(pd.to_datetime("2000-01-01"), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.to_datetime(1), pd.Timestamp), pd.Timestamp)
    check(assert_type(pd.to_datetime(1.5), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.to_datetime(dt.datetime(2000, 1, 1)), pd.Timestamp), pd.Timestamp
    )
    check(assert_type(pd.to_datetime(dt.date(2000, 1, 1)), pd.Timestamp), pd.Timestamp)
    check(
        assert_type(pd.to_datetime(np.datetime64("2000-01-01")), pd.Timestamp),
        pd.Timestamp,
    )


def test_to_datetime_series() -> None:
    s = pd.Series(["2000-01-01", "2000-01-02"])
    check(assert_type(pd.to_datetime(s), "TimestampSeries"), pd.Series)
    d: FulldatetimeDict = {
        "year": [2000, 2000, 2000],
        "month": [1, 1, 1],
        "day": [1, 2, 3],
    }
    df = pd.DataFrame(d)
    d_ex: FulldatetimeDict = {
        "year": [2000, 2000, 2000],
        "month": [1, 1, 1],
        "day": [1, 2, 3],
        "hour": [1, 1, 1],
        "hours": [1, 1, 1],
        "minute": [1, 1, 1],
        "minutes": [1, 1, 1],
        "second": [1, 1, 1],
        "seconds": [1, 1, 1],
        "ms": [1, 1, 1],
        "us": [1, 1, 1],
        "ns": [1, 1, 1],
    }
    check(assert_type(pd.to_datetime(df), "TimestampSeries"), pd.Series)
    check(assert_type(pd.to_datetime(d), "TimestampSeries"), pd.Series)
    check(assert_type(pd.to_datetime(d_ex), "TimestampSeries"), pd.Series)


def test_to_datetime_array() -> None:
    check(assert_type(pd.to_datetime([1, 2, 3]), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(pd.to_datetime([1.5, 2.5, 3.5]), pd.DatetimeIndex), pd.DatetimeIndex
    )
    check(
        assert_type(
            pd.to_datetime(
                [
                    dt.datetime(2000, 1, 1),
                    dt.datetime(2000, 1, 2),
                    dt.datetime(2000, 1, 3),
                ]
            ),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    check(assert_type(pd.to_datetime((1, 2, 3)), pd.DatetimeIndex), pd.DatetimeIndex)
    check(
        assert_type(pd.to_datetime((1.5, 2.5, 3.5)), pd.DatetimeIndex), pd.DatetimeIndex
    )
    check(
        assert_type(
            pd.to_datetime(
                (
                    dt.datetime(2000, 1, 1),
                    dt.datetime(2000, 1, 2),
                    dt.datetime(2000, 1, 3),
                )
            ),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(("2000-01-01", "2000-01-02", "2000-01-03")), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(
                np.array(
                    [
                        np.datetime64("2000-01-01"),
                        np.datetime64("2000-01-02"),
                        np.datetime64("2000-01-03"),
                    ]
                )
            ),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(np.array(["2000-01-01", "2000-01-02", "2000-01-03"])),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(pd.to_datetime(np.array([1, 2, 3])), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    pd.to_datetime(
        pd.Index([2451544.5, 2451545.5, 2451546.5]),
        unit="D",
        origin="julian",
    )
    check(
        assert_type(
            pd.to_datetime(pd.Index([1, 2, 3]), origin="unix"), pd.DatetimeIndex
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(pd.to_datetime(pd.Index([1, 2, 3]), origin=4), pd.DatetimeIndex),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(pd.Index([1, 2, 3]), origin=pd.Timestamp("1999-12-31")),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(pd.Index([1, 2, 3]), origin=np.datetime64("1999-12-31")),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(pd.Index([1, 2, 3]), origin=dt.datetime(1999, 12, 31)),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )
    check(
        assert_type(
            pd.to_datetime(pd.Index([1, 2, 3]), origin=dt.date(1999, 12, 31)),
            pd.DatetimeIndex,
        ),
        pd.DatetimeIndex,
    )


def test_timedelta_range() -> None:
    check(
        assert_type(
            pd.timedelta_range(
                pd.Timedelta(1, unit="D"), pd.Timedelta(10, unit="D"), periods=10
            ),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range(dt.timedelta(1), dt.timedelta(10), periods=10),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range(
                np.timedelta64(86400000000000),
                np.timedelta64(864000000000000),
                periods=10,
            ),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range("1 day", "10 days", periods=10), pd.TimedeltaIndex
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range(
                np.int64(86400000000000), np.int64(864000000000000), periods=10
            ),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range(86400000000000, 864000000000000, periods=10),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range(86400000000000.0, 864000000000000.0, periods=10),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range("1 day", "10 days", freq=pd.Timedelta("2 days")),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )
    check(
        assert_type(
            pd.timedelta_range("1 day", "10 days", freq=dt.timedelta(days=2)),
            pd.TimedeltaIndex,
        ),
        pd.TimedeltaIndex,
    )


def test_dateoffset_freqstr() -> None:
    offset = DateOffset(minutes=10)
    check(assert_type(offset.freqstr, str), str)


def test_timedelta64_and_arithmatic_operator() -> None:
    s1 = pd.Series(data=pd.date_range("1/1/2020", "2/1/2020"))
    s2 = pd.Series(data=pd.date_range("1/1/2021", "2/1/2021"))
    s3 = s2 - s1
    td = np.timedelta64(1, "M")
    check(assert_type((s1 - td), "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type((s1 + td), "TimestampSeries"), pd.Series, pd.Timestamp)
    check(assert_type((s3 - td), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type((s3 + td), "TimedeltaSeries"), pd.Series, pd.Timedelta)
    check(assert_type((s3 / td), "pd.Series[float]"), pd.Series, float)
    if TYPE_CHECKING_INVALID_USAGE:
        r1 = s1 * td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        r2 = s1 / td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]
        r3 = s3 * td  # type: ignore[operator] # pyright: ignore[reportGeneralTypeIssues]


def test_timedeltaseries_add_timestampseries() -> None:
    tds = pd.Series(pd.timedelta_range(start="1 day", periods=10))
    tss = pd.Series(pd.date_range(start="2012-01-01", periods=10, freq="W-MON"))
    plus = tds + tss
    check(assert_type(plus, "TimestampSeries"), pd.Series, pd.Timestamp)


def test_mean_median_std() -> None:
    s = pd.Series([pd.Timedelta("1 ns"), pd.Timedelta("2 ns"), pd.Timedelta("3 ns")])
    check(assert_type(s.mean(), pd.Timedelta), pd.Timedelta)
    check(assert_type(s.median(), pd.Timedelta), pd.Timedelta)
    check(assert_type(s.std(), pd.Timedelta), pd.Timedelta)

    s2 = pd.Series(
        [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-02"),
            pd.Timestamp("2021-01-03"),
        ]
    )
    check(assert_type(s2.mean(), pd.Timestamp), pd.Timestamp)
    check(assert_type(s2.median(), pd.Timestamp), pd.Timestamp)
    check(assert_type(s2.std(), pd.Timedelta), pd.Timedelta)


def test_timestamp_strptime_fails():
    if TYPE_CHECKING_INVALID_USAGE:
        assert_never(
            pd.Timestamp.strptime(
                "2023-02-16",  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]
                "%Y-%M-%D",  # type: ignore[arg-type] # pyright: ignore[reportGeneralTypeIssues]
            )
        )


def test_weekofmonth_init():
    # GH 629
    check(
        assert_type(
            pd.offsets.WeekOfMonth(n=1, week=1, weekday=1, normalize=True),
            pd.offsets.WeekOfMonth,
        ),
        pd.offsets.WeekOfMonth,
    )
