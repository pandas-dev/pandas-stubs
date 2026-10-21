from __future__ import annotations

from typing import assert_type

import pandas as pd

from tests import check
from tests._typing import np_1darray_bool


def test_interval_index_with_base_index() -> None:
    values = pd.IntervalIndex.from_breaks([0, 1, 2])
    other: pd.Index[pd.Interval[int]] = values

    check(assert_type(values < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(values <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(values > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(values >= other, np_1darray_bool), np_1darray_bool)
