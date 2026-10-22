from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[bool]:
    """Left operand"""
    lo = pd.Series([True, True, False])
    return check(assert_type(lo, "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_py_scalar(left: pd.Series[bool]) -> None:
    """Test pd.Series[bool] <comparison> py scalar"""
    # GH 175
    b = True

    check(assert_type(left < b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= b, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(b), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(b), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(b), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(b), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_index(left: pd.Series[bool]) -> None:
    index = pd.Index(left)

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


def test_relops_pd_series(left: pd.Series[bool]) -> None:
    other = left.copy()

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)
