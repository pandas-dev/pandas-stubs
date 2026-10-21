from __future__ import annotations

from datetime import date
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[date]:
    """Left operand"""
    lo = pd.Series([date(2022, 7, 31)], dtype=object)
    return check(assert_type(lo, "pd.Series[date]"), pd.Series, date)


def test_relops_py_scalar(left: pd.Series[date]) -> None:
    """Test pd.Series[date] <comparison> py scalar"""
    # GH 175
    dtd = date(2022, 7, 31)

    check(assert_type(left > dtd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < dtd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= dtd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= dtd, "pd.Series[bool]"), pd.Series, np.bool_)
