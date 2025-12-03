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


def test_construction() -> None:
    check(assert_type(array([1]), IntegerArray), IntegerArray)
    check(assert_type(array([1, NA]), IntegerArray), IntegerArray)
