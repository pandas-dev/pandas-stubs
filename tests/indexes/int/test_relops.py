from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check
from tests._typing import np_1darray_bool


@pytest.fixture
def left() -> pd.Index[int]:
    """Left operand"""
    lo = pd.Index([1, 2, 3])
    return check(assert_type(lo, "pd.Index[int]"), pd.Index, np.integer)


def test_relops_py_scalar(left: pd.Index[int]) -> None:
    # GH 265
    scalar = 2

    check(assert_type(left < scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= scalar, np_1darray_bool), np_1darray_bool)

    check(assert_type(scalar < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar >= left, np_1darray_bool), np_1darray_bool)


def test_relops_pd_index(left: pd.Index[int]) -> None:
    other = left.copy()

    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)
