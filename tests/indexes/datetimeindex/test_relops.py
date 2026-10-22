from __future__ import annotations

from typing import assert_type

import pandas as pd
import pytest

from tests import check
from tests._typing import np_1darray_bool


@pytest.fixture
def left() -> pd.DatetimeIndex:
    """Left operand"""
    lo = pd.date_range("2022-01-01", "2022-01-31", freq="D")
    return check(assert_type(lo, "pd.DatetimeIndex"), pd.DatetimeIndex)


def test_relops_pd_scalar(left: pd.DatetimeIndex) -> None:
    scalar = pd.Timestamp("2022-01-02")

    check(assert_type(left < scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > scalar, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= scalar, np_1darray_bool), np_1darray_bool)

    check(assert_type(scalar < left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar <= left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar > left, np_1darray_bool), np_1darray_bool)
    check(assert_type(scalar >= left, np_1darray_bool), np_1darray_bool)


def test_relops_pd_index(left: pd.DatetimeIndex) -> None:
    other = left.copy()

    check(assert_type(left < other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left <= other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left > other, np_1darray_bool), np_1darray_bool)
    check(assert_type(left >= other, np_1darray_bool), np_1darray_bool)
