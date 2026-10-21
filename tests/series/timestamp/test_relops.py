from __future__ import annotations

from datetime import datetime
from typing import assert_type

import numpy as np
import pandas as pd
import pytest

from tests import check


@pytest.fixture
def left() -> pd.Series[pd.Timestamp]:
    """Left operand"""
    lo = pd.Series([pd.Timestamp("2022-07-31 08:35:12")])
    return check(assert_type(lo, "pd.Series[pd.Timestamp]"), pd.Series, pd.Timestamp)


def test_relops_py_scalar(left: pd.Series[pd.Timestamp]) -> None:
    """Test pd.Series[pd.Timestamp] <comparison> py scalar"""
    # GH 175
    dtdt = datetime(2022, 7, 31, 8, 32, 21)
    ts = pd.Timestamp("2022-07-31 08:35:12")

    check(assert_type(left > dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= dtdt, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left > ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= ts, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= ts, "pd.Series[bool]"), pd.Series, np.bool_)
