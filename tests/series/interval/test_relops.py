from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[pd.Interval[int]]:
    """Left operand"""
    lo = pd.Series(pd.IntervalIndex.from_breaks([0, 1, 2]))
    return check(assert_type(lo, "pd.Series[pd.Interval[int]]"), pd.Series, pd.Interval)


def test_relops_pd_index(left: pd.Series[pd.Interval[int]]) -> None:
    index = pd.IntervalIndex(left.to_list())

    check(assert_type(left < index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > index, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= index, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(index < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(index >= left, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(index), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(index), "pd.Series[bool]"), pd.Series, np.bool_)
