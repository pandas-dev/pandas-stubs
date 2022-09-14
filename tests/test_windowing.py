import numpy as np
from pandas import (
    DataFrame,
    Series,
    date_range,
)
from typing_extensions import assert_type

from tests import check

IDX = date_range("1/1/2000", periods=700, freq="D")
S = Series(np.random.standard_normal(700))
DF = DataFrame({"col1": S, "col2": S})


def test_rolling():
    check(assert_type(S.rolling(10).mean(), Series), Series)
    check(assert_type(DF.rolling(10).mean(), DataFrame), DataFrame)


def test_expanding():
    check(assert_type(S.expanding().mean(), Series), Series)
    check(assert_type(DF.expanding().mean(), DataFrame), DataFrame)


def test_ewm():
    check(assert_type(S.ewm(span=10).mean(), Series), Series)
    check(assert_type(DF.ewm(span=10).mean(), DataFrame), DataFrame)
