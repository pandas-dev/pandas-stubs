from __future__ import annotations

from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[float]:
    """Left operand"""
    lo = pd.Series([1.0, 2.0, 3.0])
    return check(assert_type(lo, "pd.Series[float]"), pd.Series, np.floating)


def test_relops_py_scalar(left: pd.Series[float]) -> None:
    """Test pd.Series[float] <comparison> py scalar"""
    # GH 175
    f: float = 3.14

    check(assert_type(left > f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= f, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= f, "pd.Series[bool]"), pd.Series, np.bool_)


def test_relops_mixed_numeric(left: pd.Series[float]) -> None:
    """Test pd.Series[float] <comparison> int scalar"""
    # GH 1903
    i: int = 10

    check(assert_type(left > i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= i, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i > left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i < left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i <= left, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(i >= left, "pd.Series[bool]"), pd.Series, np.bool_)
