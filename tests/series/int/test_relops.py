from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[int]:
    """Left operand"""
    lo = pd.Series([1, 2, 3])
    return check(assert_type(lo, "pd.Series[int]"), pd.Series, np.integer)


def test_relops_py_scalar(left: pd.Series[int]) -> None:
    """Test pd.Series[int] <comparison> py scalar"""
    # GH 175
    i: int = 10

    check(assert_type(left > i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= i, "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_mixed_numeric(left: pd.Series[int]) -> None:
    """Test pd.Series[int] <comparison> float scalar"""
    # GH 1903
    f: float = 3.14

    check(assert_type(left > f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(f >= left, "pd.Series[bool]"), pd.Series, np.bool_)
