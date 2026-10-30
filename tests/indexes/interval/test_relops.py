from __future__ import annotations

from typing import assert_type

import pandas as pd
import pytest

from tests import check
from tests._typing import np_1darray_bool


@pytest.fixture
def left() -> pd.IntervalIndex[pd.Interval[int]]:
    """Left operand"""
    lo = pd.IntervalIndex.from_breaks([0, 1, 2])
    return check(
        assert_type(lo, "pd.IntervalIndex[pd.Interval[int]]"), pd.IntervalIndex
    )


def test_relops_pd_scalar(left: pd.IntervalIndex[pd.Interval[int]]) -> None:
    scalar = pd.Interval(0, 1)

    check(assert_type(left < scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= scalar, np_1darray_bool), np_1darray_bool)

    check(assert_type(scalar < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar >= left, np_1darray_bool), np_1darray_bool)


def test_relops_pd_index_base(
    left: pd.IntervalIndex[pd.Interval[int]],
) -> None:
    other: pd.Index[pd.Interval[int]] = left

    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)


def test_relops_pd_index(left: pd.IntervalIndex[pd.Interval[int]]) -> None:
    other = left.copy()

    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)
