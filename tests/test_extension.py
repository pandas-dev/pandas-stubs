import decimal

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
