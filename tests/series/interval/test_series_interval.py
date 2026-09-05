from __future__ import annotations

from typing import assert_type

import pandas as pd

from tests import check


def test_types_ffill() -> None:
    s1 = pd.interval_range(0, 2).to_series()
    check(
        assert_type(s1.ffill(), "pd.Series[pd.Interval[int]]"),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(s1.ffill(inplace=False), "pd.Series[pd.Interval[int]]"),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(
            s1.ffill(inplace=False, limit_area="inside"), "pd.Series[pd.Interval[int]]"
        ),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(s1.ffill(inplace=True), "pd.Series[pd.Interval[int]]"),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(
            s1.ffill(inplace=True, limit_area="outside"), "pd.Series[pd.Interval[int]]"
        ),
        pd.Series,
        pd.Interval,
    )


def test_types_bfill() -> None:
    s1 = pd.interval_range(0, 2).to_series()
    check(
        assert_type(s1.bfill(), "pd.Series[pd.Interval[int]]"), pd.Series, pd.Interval
    )
    check(
        assert_type(s1.bfill(inplace=False), "pd.Series[pd.Interval[int]]"),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(
            s1.bfill(inplace=False, limit_area="inside"), "pd.Series[pd.Interval[int]]"
        ),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(s1.bfill(inplace=True), "pd.Series[pd.Interval[int]]"),
        pd.Series,
        pd.Interval,
    )
    check(
        assert_type(
            s1.bfill(inplace=True, limit_area="outside"), "pd.Series[pd.Interval[int]]"
        ),
        pd.Series,
        pd.Interval,
    )
