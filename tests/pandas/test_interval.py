# flake8: noqa: F841
from typing import TYPE_CHECKING
import pandas as pd

from . import (
    check_bool_result,
    check_timedelta_result,
    check_timestamp_result,
    check_interval_result,
    check_int_result,
    check_float_result,
)


def test_interval_init() -> None:
    i1: pd.Interval = pd.Interval(1, 2, closed="both")
    i2: pd.Interval = pd.Interval(1, right=2, closed="right")
    i3: pd.Interval = pd.Interval(left=1, right=2, closed="left")


def test_interval_arithmetic() -> None:
    i1: pd.Interval = pd.Interval(1, 2, closed="both")
    i2: pd.Interval = pd.Interval(1, right=2, closed="right")

    i3: pd.Interval = i1 + 1
    i4: pd.Interval = i1 - 1
    i5: pd.Interval = i1 * 2
    i6: pd.Interval = i1 / 2
    i7: pd.Interval = i1 // 2


def test_max_intervals() -> None:
    i1 = pd.Interval(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"), closed="both")
    i2 = pd.Interval(pd.Timestamp("2000-01-01T12:00:00"), pd.Timestamp("2000-01-02"), closed="both")
    print(max(i1.left, i2.left))


def test_interval_length() -> None:
    i1 = pd.Interval(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"), closed="both")
    check_timedelta_result(i1.length)
    check_timestamp_result(i1.left)
    check_timestamp_result(i1.right)
    check_timestamp_result(i1.mid)
    i1.length.total_seconds()
    inres = pd.Timestamp("2001-01-02") in i1
    check_bool_result(inres)
    idres = i1 + pd.Timedelta(seconds=20)

    check_interval_result(idres, pd.Timestamp)
    if TYPE_CHECKING:
        20 in i1  # type: ignore
        i1 + pd.Timestamp("2000-03-03")  # type: ignore
        i1 * 3  # type: ignore
        i1 * pd.Timedelta(seconds=20)  # type: ignore

    i2 = pd.Interval(10, 20)
    check_int_result(i2.length)
    check_int_result(i2.left)
    check_int_result(i2.right)
    check_float_result(i2.mid)

    i2inres = 15 in i2
    check_bool_result(i2inres)
    check_interval_result(i2 + 3, int)
    check_interval_result(i2 + 3.2, float)
    check_interval_result(i2 * 4, int)
    check_interval_result(i2 * 4.2, float)

    if TYPE_CHECKING:
        pd.Timestamp("2001-01-02") in i2  # type: ignore
        i2 + pd.Timedelta(seconds=20)  # type: ignore

    i3 = pd.Interval(13.2, 19.5)
    check_float_result(i3.length)
    check_float_result(i3.left)
    check_float_result(i3.right)
    check_float_result(i3.mid)

    i3inres = 15.4 in i3
    check_bool_result(i3inres)
    check_interval_result(i3 + 3, float)
    check_interval_result(i3 * 3, float)
    if TYPE_CHECKING:
        pd.Timestamp("2001-01-02") in i3  # type: ignore
        i3 + pd.Timedelta(seconds=20)  # type: ignore
