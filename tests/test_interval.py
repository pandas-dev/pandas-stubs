# flake8: noqa: F841
# pyright: reportUnusedExpression = false
from typing import TYPE_CHECKING

import pandas as pd
from typing_extensions import assert_type


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
    i1 = pd.Interval(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-02"), closed="both"
    )
    i2 = pd.Interval(
        pd.Timestamp("2000-01-01T12:00:00"), pd.Timestamp("2000-01-02"), closed="both"
    )
    print(max(i1.left, i2.left))


def test_interval_length() -> None:
    i1 = pd.Interval(
        pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-03"), closed="both"
    )
    assert_type(i1.length, "pd.Timedelta")
    assert_type(i1.left, "pd.Timestamp")
    assert_type(i1.right, "pd.Timestamp")
    assert_type(i1.mid, "pd.Timestamp")
    i1.length.total_seconds()
    inres = pd.Timestamp("2001-01-02") in i1
    assert_type(inres, "bool")
    idres = i1 + pd.Timedelta(seconds=20)

    assert_type(idres, "pd.Interval[pd.Timestamp]")
    if TYPE_CHECKING:
        20 in i1
        i1 + pd.Timestamp("2000-03-03")  # type: ignore[operator]
        i1 * 3  # type: ignore[operator]
        i1 * pd.Timedelta(seconds=20)  # type: ignore[operator]

    i2 = pd.Interval(10, 20)
    assert_type(i2.length, "int")
    assert_type(i2.left, "int")
    assert_type(i2.right, "int")
    assert_type(i2.mid, "float")

    i2inres = 15 in i2
    assert_type(i2inres, "bool")
    assert_type(i2 + 3, "pd.Interval[int]")
    assert_type(i2 + 3.2, "pd.Interval[float]")
    assert_type(i2 * 4, "pd.Interval[int]")
    assert_type(i2 * 4.2, "pd.Interval[float]")

    if TYPE_CHECKING:
        pd.Timestamp("2001-01-02") in i2  # pyright: ignore[reportGeneralTypeIssues]
        i2 + pd.Timedelta(seconds=20)  # pyright: ignore[reportGeneralTypeIssues]

    i3 = pd.Interval(13.2, 19.5)
    assert_type(i3.length, "float")
    assert_type(i3.left, "float")
    assert_type(i3.right, "float")
    assert_type(i3.mid, "float")

    i3inres = 15.4 in i3
    assert_type(i3inres, "bool")
    assert_type(i3 + 3, "pd.Interval[float]")
    assert_type(i3 * 3, "pd.Interval[float]")
    if TYPE_CHECKING:
        pd.Timestamp("2001-01-02") in i3  # pyright: ignore[reportGeneralTypeIssues]
        i3 + pd.Timedelta(seconds=20)  # pyright: ignore[reportGeneralTypeIssues]
