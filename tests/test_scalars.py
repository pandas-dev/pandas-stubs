from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from typing_extensions import assert_type

from tests import check


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
