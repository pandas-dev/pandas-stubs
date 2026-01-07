from pandas.core.arrays.integer import IntegerArray
from pandas.core.construction import array
from typing_extensions import assert_type

from pandas._libs.missing import NA

from tests import check


def test_cumul_int64dtype() -> None:
    arr = array([1, NA, 2])

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
