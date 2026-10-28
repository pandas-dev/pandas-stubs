from __future__ import annotations

from typing import assert_type

import pandas as pd

from tests import check
from tests._typing import np_1darray_bool


def test_relops_ordered_categorical() -> None:
    left = pd.CategoricalIndex(["a", "b"], categories=["a", "b"], ordered=True)
    other = left.copy()

    check(assert_type(left < "b", np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= "b", np_1darray_bool), np_1darray_bool)
    check(assert_type(left > "b", np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= "b", np_1darray_bool), np_1darray_bool)

    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)
