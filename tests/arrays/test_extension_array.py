# Test common ExtensionArray methods
import numpy as np
import pandas as pd
from pandas.core.arrays.integer import IntegerArray
from pandas.core.construction import array
from typing_extensions import assert_type

from tests import check


def test_ea_common() -> None:
    # Note: `ExtensionArray` is abstract, so we use `IntegerArray` for the tests.
    arr = array([1, 2, 3])

    check(assert_type(arr.repeat(1), IntegerArray), IntegerArray)
    check(assert_type(arr.repeat(arr), IntegerArray), IntegerArray)
    check(assert_type(arr.repeat(np.array([1, 2, 3])), IntegerArray), IntegerArray)
    check(
        assert_type(arr.repeat(repeats=pd.Series([1, 2, 3])), IntegerArray),
        IntegerArray,
    )
    check(assert_type(arr.repeat(pd.Index([1, 2, 3])), IntegerArray), IntegerArray)
    check(assert_type(arr.repeat([1, 2, 3]), IntegerArray), IntegerArray)

    check(assert_type(arr.unique(), IntegerArray), IntegerArray)
    check(assert_type(arr.dropna(), IntegerArray), IntegerArray)
    check(assert_type(arr.take([1, 0, 2]), IntegerArray), IntegerArray)
    check(
        assert_type(arr.take([1, 0, 2], allow_fill=True, fill_value=-1), IntegerArray),
        IntegerArray,
    )
    check(assert_type(arr.ravel(), IntegerArray), IntegerArray)
