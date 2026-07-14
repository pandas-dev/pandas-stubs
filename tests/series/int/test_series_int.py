from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd

from tests import check


def test_types_ffill() -> None:
    s1 = pd.Series([1, 2, 3])
    check(assert_type(s1.ffill(), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s1.ffill(inplace=False), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s1.ffill(inplace=False, limit_area="inside"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(assert_type(s1.ffill(inplace=True), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s1.ffill(inplace=True, limit_area="outside"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


def test_types_bfill() -> None:
    s1 = pd.Series([1, 2, 3])
    check(assert_type(s1.bfill(), "pd.Series[int]"), pd.Series, np.integer)
    check(assert_type(s1.bfill(inplace=False), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s1.bfill(inplace=False, limit_area="inside"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(assert_type(s1.bfill(inplace=True), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(s1.bfill(inplace=True, limit_area="outside"), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
