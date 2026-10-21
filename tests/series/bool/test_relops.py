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
    b: bool = True

    check(assert_type(left > b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= b, "pd.Series[bool]"), pd.Series, np.bool_)
