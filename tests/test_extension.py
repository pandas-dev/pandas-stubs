import decimal

import pandas as pd
from typing_extensions import assert_type

from tests import check
from tests.extension.decimal.array import (
    DecimalArray,
    DecimalDtype,
)


def test_constructor() -> None:
    arr = DecimalArray([decimal.Decimal("1.0"), decimal.Decimal("2.0")])

    check(assert_type(arr, DecimalArray), DecimalArray, decimal.Decimal)
    check(assert_type(arr.dtype, DecimalDtype), DecimalDtype)


def test_tolist() -> None:
    data = {"State": "Texas", "Population": 2000000, "GDP": "2T"}
    s = pd.Series(data)
    data1 = [1, 2, 3]
    s1 = pd.Series(data1)
    check(assert_type(s.array.tolist(), list), list)
    check(assert_type(s1.array.tolist(), list), list)
    check(assert_type(pd.array([1, 2, 3]).tolist(), list), list)
