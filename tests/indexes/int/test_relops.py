from __future__ import annotations

from typing import assert_type

import pandas as pd

from tests import check
from tests._typing import np_1darray_bool


def test_int_index_relops() -> None:
    # GH 265
    ind = pd.Index([1, 2, 3])

    check(assert_type(ind <= 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(ind < 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(ind >= 2, np_1darray_bool), np_1darray_bool)
    check(assert_type(ind > 2, np_1darray_bool), np_1darray_bool)
