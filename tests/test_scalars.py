from __future__ import annotations

from typing import Literal

import pandas as pd
from typing_extensions import assert_type

from tests import check


def test_interval() -> None:
    i0 = pd.Interval(0, 1, closed="left")
    i1 = pd.Interval(0.0, 1.0, closed="right")
    i2 = pd.Interval(
        pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-02"), closed="both"
    )
    i3 = pd.Interval(pd.Timedelta("1 days"), pd.Timedelta("2 days"), closed="neither")
    check(assert_type(i0, "pd.Interval[int]"), pd.Interval)
    check(assert_type(i1, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i2, "pd.Interval[pd.Timestamp]"), pd.Interval)
    check(assert_type(i3, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0.closed, Literal["left", "right", "both", "neither"]), str)
    check(assert_type(i0.closed_left, bool), bool)
    check(assert_type(i0.closed_right, bool), bool)
    check(assert_type(i0.is_empty, bool), bool)
    check(assert_type(i0.left, int), int)
    check(assert_type(i0.length, int), int)
    check(assert_type(i0.mid, float), float)
    check(assert_type(i0.open_left, bool), bool)
    check(assert_type(i0.open_right, bool), bool)
    check(assert_type(i0.right, int), int)

    check(assert_type(i1.closed, Literal["left", "right", "both", "neither"]), str)
    check(assert_type(i1.closed_left, bool), bool)
    check(assert_type(i1.closed_right, bool), bool)
    check(assert_type(i1.is_empty, bool), bool)
    check(assert_type(i1.left, float), float)
    check(assert_type(i1.length, float), float)
    check(assert_type(i1.mid, float), float)
    check(assert_type(i1.open_left, bool), bool)
    check(assert_type(i1.open_right, bool), bool)
    check(assert_type(i1.right, float), float)

    check(assert_type(i2.closed, Literal["left", "right", "both", "neither"]), str)
    check(assert_type(i2.closed_left, bool), bool)
    check(assert_type(i2.closed_right, bool), bool)
    check(assert_type(i2.is_empty, bool), bool)
    check(assert_type(i2.left, pd.Timestamp), pd.Timestamp)
    check(assert_type(i2.length, pd.Timedelta), pd.Timedelta)
    check(assert_type(i2.mid, pd.Timestamp), pd.Timestamp)
    check(assert_type(i2.open_left, bool), bool)
    check(assert_type(i2.open_right, bool), bool)
    check(assert_type(i2.right, pd.Timestamp), pd.Timestamp)

    check(assert_type(i3.closed, Literal["left", "right", "both", "neither"]), str)
    check(assert_type(i3.closed_left, bool), bool)
    check(assert_type(i3.closed_right, bool), bool)
    check(assert_type(i3.is_empty, bool), bool)
    check(assert_type(i3.left, pd.Timedelta), pd.Timedelta)
    check(assert_type(i3.length, pd.Timedelta), pd.Timedelta)
    check(assert_type(i3.mid, pd.Timedelta), pd.Timedelta)
    check(assert_type(i3.open_left, bool), bool)
    check(assert_type(i3.open_right, bool), bool)
    check(assert_type(i3.right, pd.Timedelta), pd.Timedelta)

    check(assert_type(i0.overlaps(pd.Interval(0.5, 1.5, closed="left")), bool), bool)
    check(assert_type(i0.overlaps(pd.Interval(2, 3, closed="left")), bool), bool)

    check(assert_type(i1.overlaps(pd.Interval(0.5, 1.5, closed="left")), bool), bool)
    check(assert_type(i1.overlaps(pd.Interval(2, 3, closed="left")), bool), bool)
    ts1 = pd.Timestamp(year=2017, month=1, day=1)
    ts2 = pd.Timestamp(year=2017, month=1, day=2)
    check(assert_type(i2.overlaps(pd.Interval(ts1, ts2, closed="left")), bool), bool)
    td1 = pd.Timedelta(days=1)
    td2 = pd.Timedelta(days=3)
    check(assert_type(i3.overlaps(pd.Interval(td1, td2, closed="left")), bool), bool)

    check(assert_type(i0 * 3, "pd.Interval[int]"), pd.Interval)
    check(assert_type(i1 * 3, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i3 * 3, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0 * 3.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i1 * 3.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i3 * 3.5, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(3 * i0, "pd.Interval[int]"), pd.Interval)
    check(assert_type(3 * i1, "pd.Interval[float]"), pd.Interval)
    check(assert_type(3 * i3, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(3.5 * i0, "pd.Interval[float]"), pd.Interval)
    check(assert_type(3.5 * i1, "pd.Interval[float]"), pd.Interval)
    check(assert_type(3.5 * i3, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0 / 3, "pd.Interval[int]"), pd.Interval)
    check(assert_type(i1 / 3, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i3 / 3, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0 / 3.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i1 / 3.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i3 / 3.5, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0 // 3, "pd.Interval[int]"), pd.Interval)
    check(assert_type(i1 // 3, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i3 // 3, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0 // 3.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i1 // 3.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i3 // 3.5, "pd.Interval[pd.Timedelta]"), pd.Interval)

    check(assert_type(i0 - 1, "pd.Interval[int]"), pd.Interval)
    check(assert_type(i1 - 1, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(i2 - pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
    )
    check(
        assert_type(i3 - pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
    )

    check(assert_type(i0 - 1.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i1 - 1.5, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(i2 - pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
    )
    check(
        assert_type(i3 - pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
    )

    check(assert_type(i0 + 1, "pd.Interval[int]"), pd.Interval)
    check(assert_type(i1 + 1, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(i2 + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
    )
    check(
        assert_type(i3 + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
    )

    check(assert_type(i0 + 1.5, "pd.Interval[float]"), pd.Interval)
    check(assert_type(i1 + 1.5, "pd.Interval[float]"), pd.Interval)
    check(
        assert_type(i2 + pd.Timedelta(days=1), "pd.Interval[pd.Timestamp]"), pd.Interval
    )
    check(
        assert_type(i3 + pd.Timedelta(days=1), "pd.Interval[pd.Timedelta]"), pd.Interval
    )

    check(assert_type(0.5 in i0, bool), bool)
    check(assert_type(1 in i0, bool), bool)
    check(assert_type(1 in i1, bool), bool)
    check(assert_type(pd.Timestamp("2000-1-1") in i2, bool), bool)
    check(assert_type(pd.Timedelta(days=1) in i3, bool), bool)

    check(assert_type(hash(i0), int), int)
    check(assert_type(hash(i1), int), int)
    check(assert_type(hash(i2), int), int)
    check(assert_type(hash(i3), int), int)
