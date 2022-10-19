from __future__ import annotations

import datetime as dt
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from pandas._libs.tslibs import (
    BaseOffset,
    NaTType,
)

if TYPE_CHECKING:
    from pandas.core.series import (
        PeriodSeries,
        TimedeltaSeries,
    )

    from pandas._typing import np_ndarray_bool
else:
    PeriodSeries = TimedeltaSeries = np_ndarray_bool = Any

from tests import check

from pandas.tseries.offsets import Day


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


def test_periof_add_subtract() -> None:
    p = pd.Period("2012-1-1", freq="D")

    as_pd_td = pd.Timedelta(1, "D")
    as_dt_td = dt.timedelta(days=1)
    as_np_td = np.timedelta64(1, "D")
    as_np_i64 = np.int64(1)
    as_int = int(1)
    as_period_index = pd.period_range("2012-1-1", periods=10, freq="D")
    as_period = pd.Period("2012-1-1", freq="D")
    scale = 24 * 60 * 60 * 10**9
    as_td_series = pd.Series(pd.timedelta_range(scale, scale, freq="D"))
    check(assert_type(as_td_series, TimedeltaSeries), pd.Series)
    as_period_series = pd.Series(as_period_index)
    as_timedelta_idx = pd.timedelta_range(scale, scale, freq="D")
    as_nat = pd.NaT

    check(assert_type(p + as_pd_td, pd.Period), pd.Period)
    check(assert_type(p + as_dt_td, pd.Period), pd.Period)
    check(assert_type(p + as_np_td, pd.Period), pd.Period)
    check(assert_type(p + as_np_i64, pd.Period), pd.Period)
    check(assert_type(p + as_int, pd.Period), pd.Period)
    check(assert_type(p + p.freq, pd.Period), pd.Period)
    check(assert_type(p + (p - as_period_index), pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p + as_td_series, PeriodSeries), pd.Series, pd.Period)
    check(assert_type(p + as_timedelta_idx, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p + as_nat, NaTType), NaTType)
    das8 = cast(TimedeltaSeries, (as_period_series - as_period_series))
    check(assert_type(p + das8, PeriodSeries), pd.Series, pd.Period)
    check(assert_type(p - as_pd_td, pd.Period), pd.Period)
    check(assert_type(p - as_dt_td, pd.Period), pd.Period)
    check(assert_type(p - as_np_td, pd.Period), pd.Period)
    check(assert_type(p - as_np_i64, pd.Period), pd.Period)
    check(assert_type(p - as_int, pd.Period), pd.Period)
    check(assert_type(p - as_period_index, pd.Index), pd.Index)
    check(assert_type(p - as_period, BaseOffset), Day)
    check(assert_type(p - as_td_series, PeriodSeries), pd.Series, pd.Period)
    check(assert_type(p - as_timedelta_idx, pd.PeriodIndex), pd.PeriodIndex)
    check(assert_type(p - as_nat, NaTType), NaTType)
    check(assert_type(p - p.freq, pd.Period), pd.Period)

    check(assert_type(as_pd_td + p, pd.Period), pd.Period)
    check(assert_type(as_dt_td + p, pd.Period), pd.Period)
    check(assert_type(as_np_td + p, pd.Period), pd.Period)
    check(assert_type(as_np_i64 + p, pd.Period), pd.Period)
    check(assert_type(as_int + p, pd.Period), pd.Period)
    check(assert_type(as_td_series + p, PeriodSeries), pd.Series, pd.Period)
    # TODO: Improve Index to not handle __add__(period)
    check(assert_type(as_timedelta_idx + p, pd.Index), pd.PeriodIndex)
    check(assert_type(as_nat + p, NaTType), NaTType)
    check(assert_type(p.freq + p, pd.Period), pd.Period)

    check(assert_type(as_period_index - p, pd.Index), pd.Index)

    check(assert_type(p.__radd__(as_pd_td), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_dt_td), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_np_td), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_np_i64), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_int), pd.Period), pd.Period)
    check(assert_type(p.__radd__(as_nat), NaTType), NaTType)
    check(assert_type(p.__radd__(p.freq), pd.Period), pd.Period)


def test_period_cmp() -> None:
    p = pd.Period("2012-1-1", freq="D")

    c_period = pd.Period("2012-1-1", freq="D")
    c_period_index = pd.period_range("2012-1-1", periods=10, freq="D")

    eq = check(assert_type(p == c_period, bool), bool)
    ne = check(assert_type(p != c_period, bool), bool)
    assert eq != ne

    eq_a = check(assert_type(p == c_period_index, np_ndarray_bool), np.ndarray)
    ne_q = check(assert_type(p != c_period_index, np_ndarray_bool), np.ndarray)
    assert (eq_a != ne_q).all()

    eq = check(assert_type(c_period == p, bool), bool)
    ne = check(assert_type(c_period != p, bool), bool)
    assert eq != ne

    eq_a = check(assert_type(c_period_index == p, np_ndarray_bool), np.ndarray)
    ne_a = check(assert_type(c_period_index != p, np_ndarray_bool), np.ndarray)
    assert (eq_a != ne_q).all()

    gt = check(assert_type(p > c_period, bool), bool)
    le = check(assert_type(p <= c_period, bool), bool)
    assert gt != le

    gt_a = check(assert_type(p > c_period_index, np_ndarray_bool), np.ndarray)
    le_a = check(assert_type(p <= c_period_index, np_ndarray_bool), np.ndarray)
    assert (gt_a != le_a).all()

    gt = check(assert_type(c_period > p, bool), bool)
    le = check(assert_type(c_period <= p, bool), bool)
    assert gt != le

    gt_a = check(assert_type(c_period_index > p, np_ndarray_bool), np.ndarray)
    le_a = check(assert_type(c_period_index <= p, np_ndarray_bool), np.ndarray)
    assert (gt_a != le_a).all()

    lt = check(assert_type(p < c_period, bool), bool)
    ge = check(assert_type(p >= c_period, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(p < c_period_index, np_ndarray_bool), np.ndarray)
    ge_a = check(assert_type(p >= c_period_index, np_ndarray_bool), np.ndarray)
    assert (lt_a != ge_a).all()

    lt = check(assert_type(c_period < p, bool), bool)
    ge = check(assert_type(c_period >= p, bool), bool)
    assert lt != ge

    lt_a = check(assert_type(c_period_index < p, np_ndarray_bool), np.ndarray)
    ge_a = check(assert_type(c_period_index >= p, np_ndarray_bool), np.ndarray)
    assert (lt_a != ge_a).all()


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
