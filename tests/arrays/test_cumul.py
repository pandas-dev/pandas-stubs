import sys

from pandas.core.arrays.integer import IntegerArray
from typing_extensions import assert_type

from pandas._libs.missing import NA

from tests import check

if sys.version_info >= (3, 11):
    from pandas.core.construction import array
else:
    from pandas.core.construction import (
        array,  # pyright: ignore[reportUnknownVariableType]
    )


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
