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
    b: bytes = b"def"

    check(assert_type(left > b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left < b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left <= b, "pd.Series[bool]"), pd.Series, np.bool_)
    check(assert_type(left >= b, "pd.Series[bool]"), pd.Series, np.bool_)
