import pandas as pd
from pandas.core.arrays.interval import IntervalArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    itv = pd.Interval(0, 1)
    check(assert_type(pd.array([itv]), IntervalArray), IntervalArray)
    check(assert_type(pd.array([itv, None]), IntervalArray), IntervalArray)

    check(assert_type(pd.array(pd.array([itv])), IntervalArray), IntervalArray)

    check(assert_type(pd.array(pd.Index([itv])), IntervalArray), IntervalArray)

    check(assert_type(pd.array(pd.Series([itv])), IntervalArray), IntervalArray)
