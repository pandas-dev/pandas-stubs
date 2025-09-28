"""Test module for subtraction operation on Series[Period]."""

import pandas as pd
from typing_extensions import assert_type

from pandas._libs.tslibs.offsets import BaseOffset

from tests import check


def test_sub() -> None:
    """Test sub method for pd.Series[pd.Period]."""
    p = pd.Period("2012-1-1", freq="D")
    sr = pd.Series([pd.Period("2012-1-1", freq="D")])

    check(assert_type(sr - sr, "pd.Series[BaseOffset]"), pd.Series, BaseOffset)
    check(assert_type(p - sr, "pd.Series[BaseOffset]"), pd.Series, BaseOffset)
    check(assert_type(sr - p, "pd.Series[BaseOffset]"), pd.Series, BaseOffset)
    check(assert_type(sr.sub(p), "pd.Series[BaseOffset]"), pd.Series, BaseOffset)
    check(assert_type(sr.rsub(p), "pd.Series[BaseOffset]"), pd.Series, BaseOffset)
