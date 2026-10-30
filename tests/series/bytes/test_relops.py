from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[bytes]:
    """Left operand"""
    lo = pd.Series([b"1", b"2", b"3"])
    return check(assert_type(lo, "pd.Series[bytes]"), pd.Series, bytes)


def test_relops_py_scalar(left: pd.Series[bytes]) -> None:
    """Test pd.Series[bytes] <comparison> py scalar"""
    # GH 175
    b = b"def"

    check(assert_type(left < b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= b, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(b), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(b), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(b), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(b), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_numpy_array(left: pd.Series[bytes]) -> None:
    other = np.array([b"a", b"b", b"c"], dtype=np.bytes_)

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.__lt__(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__le__(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__gt__(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.__ge__(other), "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_pd_index(left: pd.Series[bytes]) -> None:
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


def test_relops_pd_series(left: pd.Series[bytes]) -> None:
    other = left.copy()

    check(assert_type(left < other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > other, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= other, "pd.Series[bool]"), pd.Series, np.bool_)

    check(assert_type(left.lt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.le(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.gt(other), "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left.ge(other), "pd.Series[bool]"), pd.Series, np.bool_)
