from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.core.arrays.interval import IntervalArray
from typing_extensions import assert_type

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import np_1darray_bool


def test_interval_init() -> None:
    check(
        assert_type(pd.Interval(1, 2, closed="both"), "pd.Interval[int]"),
        pd.Interval,
        int,
    )
    check(
        assert_type(pd.Interval(1, right=2, closed="right"), "pd.Interval[int]"),
        pd.Interval,
        int,
    )
    check(
        assert_type(pd.Interval(left=1, right=2, closed="left"), "pd.Interval[int]"),
        pd.Interval,
        int,
    )


def test_interval_arithmetic() -> None:
    i1 = pd.Interval(1, 2, closed="both")
    check(assert_type(i1, "pd.Interval[int]"), pd.Interval, int)
    check(
        assert_type(pd.Interval(1, right=2, closed="right"), "pd.Interval[int]"),
        pd.Interval,
        int,
    )

    check(assert_type(i1 + 1, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(i1 - 1, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(i1 * 2, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(i1 / 2, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(i1 // 2, "pd.Interval[int]"), pd.Interval, int)


def test_max_intervals() -> None:
    check(
        assert_type(
            pd.Interval(
                pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"), closed="both"
            ),
            "pd.Interval[pd.Timestamp]",
        ),
        pd.Interval,
        pd.Timestamp,
    )
    check(
        assert_type(
            pd.Interval(
                pd.Timestamp("2000-01-01T12:00:00"),
                pd.Timestamp("2000-01-02"),
                closed="both",
            ),
            "pd.Interval[pd.Timestamp]",
        ),
        pd.Interval,
        pd.Timestamp,
    )


def test_interval_length() -> None:
    i1 = pd.Interval(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"), closed="both"
    )
    check(assert_type(i1.length, pd.Timedelta), pd.Timedelta)
    check(assert_type(i1.left, pd.Timestamp), pd.Timestamp)
    check(assert_type(i1.right, pd.Timestamp), pd.Timestamp)
    check(assert_type(i1.mid, pd.Timestamp), pd.Timestamp)
    i1.length.total_seconds()
    inres = pd.Timestamp("2001-01-02") in i1
    check(assert_type(inres, bool), bool)
    idres = i1 + pd.Timedelta(seconds=20)

    check(assert_type(idres, "pd.Interval[pd.Timestamp]"), pd.Interval, pd.Timestamp)
    if TYPE_CHECKING_INVALID_USAGE:
        _00 = 20 in i1  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _01 = i1 + pd.Timestamp("2000-03-03")  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _02 = i1 * 3  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
        _03 = i1 * pd.Timedelta(seconds=20)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]

    i2 = pd.Interval(10, 20)
    check(assert_type(i2.length, int), int)
    check(assert_type(i2.left, int), int)
    check(assert_type(i2.right, int), int)
    check(assert_type(i2.mid, float), float)

    i2inres = 15 in i2
    check(assert_type(i2inres, bool), bool)
    check(assert_type(i2 + 3, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(i2 + 3.2, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(i2 * 4, "pd.Interval[int]"), pd.Interval, int)
    check(assert_type(i2 * 4.2, "pd.Interval[float]"), pd.Interval, float)

    if TYPE_CHECKING_INVALID_USAGE:
        _10 = pd.Timestamp("2001-01-02") in i2  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _11 = i2 + pd.Timedelta(seconds=20)  # type: ignore[type-var] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]
    i3 = pd.Interval(13.2, 19.5)
    check(assert_type(i3.length, float), float)
    check(assert_type(i3.left, float), float)
    check(assert_type(i3.right, float), float)
    check(assert_type(i3.mid, float), float)

    i3inres = 15.4 in i3
    check(assert_type(i3inres, bool), bool)
    check(assert_type(i3 + 3, "pd.Interval[float]"), pd.Interval, float)
    check(assert_type(i3 * 3, "pd.Interval[float]"), pd.Interval, float)
    if TYPE_CHECKING_INVALID_USAGE:
        _20 = pd.Timestamp("2001-01-02") in i3  # type: ignore[operator] # pyright: ignore[reportOperatorIssue]
        _21 = i3 + pd.Timedelta(seconds=20)  # type: ignore[operator] # pyright: ignore[reportOperatorIssue,reportUnknownVariableType]


def test_interval_array_contains() -> None:
    df = pd.DataFrame({"A": range(1, 10)})
    obj = pd.Interval(1, 4)
    ser = pd.Series(obj, index=df.index)
    arr = check(assert_type(ser.array, IntervalArray), IntervalArray)
    check(assert_type(arr.contains(df["A"]), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(arr.contains(3), np_1darray_bool), np_1darray_bool)
