from typing import cast

from pandas.core.arrays.integer import (
    Int64Dtype,
    IntegerArray,
)
from pandas.core.series import Series
from typing_extensions import assert_type

from pandas._libs.missing import NA

from tests import check


def test_cumul_int64dtype():
    # cast will be removed if pandas-dev/pandas-stubs#1395 is resolved
    arr = cast("Series[Int64Dtype]", Series([1, NA, 2], dtype="Int64")).array

    check(assert_type(arr._accumulate("cummin"), IntegerArray), IntegerArray)
    check(assert_type(arr._accumulate("cummax"), IntegerArray), IntegerArray)
    check(assert_type(arr._accumulate("cumsum"), IntegerArray), IntegerArray)
    check(assert_type(arr._accumulate("cumprod"), IntegerArray), IntegerArray)

    check(
        assert_type(arr._accumulate("cummin", skipna=False), IntegerArray),
        IntegerArray,
    )
    check(
        assert_type(arr._accumulate("cummax", skipna=False), IntegerArray),
        IntegerArray,
    )
    check(
        assert_type(arr._accumulate("cumsum", skipna=False), IntegerArray),
        IntegerArray,
    )
    check(
        assert_type(arr._accumulate("cumprod", skipna=False), IntegerArray),
        IntegerArray,
    )
