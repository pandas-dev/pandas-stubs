from __future__ import annotations

from datetime import timedelta
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[pd.Timedelta]:
    """Left operand"""
    lo = pd.Series([pd.Timedelta(seconds=10)])
    return check(assert_type(lo, "pd.Series[pd.Timedelta]"), pd.Series, pd.Timedelta)


def test_relops_py_scalar(left: pd.Series[pd.Timedelta]) -> None:
    """Test pd.Series[pd.Timedelta] <comparison> py scalar"""
    # GH 175
    dttd = timedelta(seconds=10)
    td = pd.Timedelta(seconds=10)

    check(assert_type(left > dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= dttd, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= td, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= td, "pd.Series[bool]"), pd.Series, np.bool_)
