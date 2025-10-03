from pandas.core.arrays.integer import IntegerArray
from pandas.core.construction import array
from typing_extensions import assert_type

from pandas._libs.missing import NA

from tests import check


def test_construction() -> None:
    check(assert_type(array([1]), IntegerArray), IntegerArray)
    check(assert_type(array([1, NA]), IntegerArray), IntegerArray)
