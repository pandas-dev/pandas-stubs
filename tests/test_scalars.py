from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Literal,
)

import numpy as np
import pandas as pd
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
    )

    from pandas._typing import np_ndarray_bool
else:
    import numpy.typing as npt

    np_ndarray_bool = npt.NDArray[np.bool_]
    PeriodSeries: TypeAlias = pd.Series
    TimedeltaSeries: TypeAlias = pd.Series
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

    check(assert_type(interval_i + 1, "pd.Interval[int]"), pd.Interval)
    check(assert_type(interval_f + 1, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(interval_ts + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(interval_td + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )

    check(assert_type(interval_i + 1.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(interval_f + 1.5, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(interval_ts + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"),
        pd.Interval,
    )
    check(
        assert_type(interval_td + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"),
        pd.Interval,
    )

    check(assert_type(0.5 in interval_i, bool), bool)
    check(assert_type(1 in interval_i, bool), bool)
    check(assert_type(1 in interval_f, bool), bool)
    check(assert_type(pd.Timestamp("2000-1-1") in interval_ts, bool), bool)
    check(assert_type(pd.Timedelta(days=1) in interval_td, bool), bool)

    check(assert_type(hash(interval_i), int), int)
    check(assert_type(hash(interval_f), int), int)
    check(assert_type(hash(interval_ts), int), int)
    check(assert_type(hash(interval_td), int), int)

    interval_index_int = pd.IntervalIndex([interval_i])
    interval_series_int = pd.Series(interval_index_int)

    check(interval_series_int >= interval_i, pd.Series, bool)
    check(interval_series_int < interval_i, pd.Series, bool)
    check(interval_series_int <= interval_i, pd.Series, bool)
    check(interval_series_int > interval_i, pd.Series, bool)

    check(interval_i >= interval_series_int, pd.Series, bool)
    check(interval_i < interval_series_int, pd.Series, bool)
    check(interval_i <= interval_series_int, pd.Series, bool)
    check(interval_i > interval_series_int, pd.Series, bool)

    check(interval_series_int == interval_i, pd.Series, bool)
    check(interval_series_int != interval_i, pd.Series, bool)

    check(
        interval_i
        == interval_series_int,  # pyright: ignore[reportUnnecessaryComparison]
        pd.Series,
        bool,
    )
    check(
        interval_i
        != interval_series_int,  # pyright: ignore[reportUnnecessaryComparison]
        pd.Series,
        bool,
    )

    check(interval_index_int >= interval_i, np.ndarray, np.bool_)
    check(interval_index_int < interval_i, np.ndarray, np.bool_)
    check(interval_index_int <= interval_i, np.ndarray, np.bool_)
    check(interval_index_int > interval_i, np.ndarray, np.bool_)

    check(interval_i >= interval_index_int, np.ndarray, np.bool_)
    check(interval_i < interval_index_int, np.ndarray, np.bool_)
    check(interval_i <= interval_index_int, np.ndarray, np.bool_)
    check(interval_i > interval_index_int, np.ndarray, np.bool_)

    check(interval_index_int == interval_i, np.ndarray, np.bool_)
    check(interval_index_int != interval_i, np.ndarray, np.bool_)

    check(
        interval_i
        == interval_index_int,  # pyright: ignore[reportUnnecessaryComparison]
        np.ndarray,
        np.bool_,
    )
    check(
        interval_i
        != interval_index_int,  # pyright: ignore[reportUnnecessaryComparison]
        np.ndarray,
        np.bool_,
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
