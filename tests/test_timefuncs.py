# flake8: noqa: F841

import datetime as dt
from typing import TYPE_CHECKING
from typing_extensions import assert_type

import pandas as pd
from pandas.testing import assert_series_equal

if TYPE_CHECKING:
    from pandas.core.series import TimedeltaSeries, TimestampSeries


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
    ts = pd.Timestamp("2022-03-05")
    s1 = pd.Series(["2022-03-05", "2022-03-06"])
    ts1 = pd.to_datetime(pd.Series(["2022-03-05", "2022-03-06"]))
    assert isinstance(ts1.iloc[0], pd.Timestamp)
    td1 = pd.to_timedelta([2, 3], "seconds")
    ts2 = pd.to_datetime(pd.Series(["2022-03-08", "2022-03-10"]))
    r1 = ts1 - ts2
    assert_type(r1, "TimedeltaSeries")
    r2 = r1 / td1
    assert_type(r2, "pd.Series[float]")
    r3 = r1 - td1
    assert_type(r3, "TimedeltaSeries")
    r4 = pd.Timedelta(5, "days") / r1
    assert_type(r4, "pd.Series[float]")
    sb = pd.Series([1, 2]) == pd.Series([1, 3])
    assert_type(sb, "pd.Series[bool]")
    r5 = sb * r1
    assert_type(r5, "TimedeltaSeries")
    r6 = r1 * 4
    assert_type(r6, "TimedeltaSeries")


def test_timestamp_dateoffset_arithmetic() -> None:
    ts = pd.Timestamp("2022-03-18")
    do = pd.DateOffset(days=366)
    r1: pd.Timestamp = ts + do


def test_datetimeindex_plus_timedelta() -> None:
    tscheck = pd.Series([pd.Timestamp("2022-03-05"), pd.Timestamp("2022-03-06")])
    dti = pd.to_datetime(["2022-03-08", "2022-03-15"])
    td_s = pd.to_timedelta(pd.Series([10, 20]), "minutes")
    dti_td_s = dti + td_s
    assert_type(dti_td_s, "TimestampSeries")
    td_dti_s = td_s + dti
    assert_type(td_dti_s, "TimestampSeries")
    tdi = pd.to_timedelta([10, 20], "minutes")
    dti_tdi_dti = dti + tdi
    assert_type(dti_tdi_dti, "pd.DatetimeIndex")
    tdi_dti_dti = tdi + dti
    assert_type(tdi_dti_dti, "pd.DatetimeIndex")
    dti_td_dti = dti + pd.Timedelta(10, "minutes")
    assert_type(dti_td_dti, "pd.DatetimeIndex")


def test_timestamp_plus_timedelta_series() -> None:
    tscheck = pd.Series([pd.Timestamp("2022-03-05"), pd.Timestamp("2022-03-06")])
    ts = pd.Timestamp("2022-03-05")
    td = pd.to_timedelta(pd.Series([10, 20]), "minutes")
    r3 = td + ts
    assert_type(r3, "TimestampSeries")
    # ignore type on next, because `tscheck` has Unknown dtype
    assert_type(r3, "TimestampSeries")


def test_timedelta_series_mult() -> None:
    df = pd.DataFrame({"x": [1, 3, 5], "y": [2, 2, 6]})
    std = (df["x"] < df["y"]) * pd.Timedelta(10, "minutes")
    assert_type(std, "TimedeltaSeries")


def test_timedelta_series_sum() -> None:
    s = pd.Series(pd.to_datetime(["04/05/2022 11:00", "04/03/2022 10:00"])) - pd.Series(
        pd.to_datetime(["04/05/2022 08:00", "04/03/2022 09:00"])
    )
    ssum = s.sum()
    ires: int = ssum.days

    sf = pd.Series([1.0, 2.2, 3.3])
    sfsum: float = sf.sum()
