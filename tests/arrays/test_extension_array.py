# Test common ExtensionArray methods
import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.integer import (
    Int32Dtype,
    IntegerArray,
)
from pandas.core.construction import array
from typing_extensions import assert_type

from pandas._typing import ArrayLike

from tests import (
    check,
    np_1darray,
)


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

    check(assert_type(arr.astype(np.dtype("int64")), np.ndarray), np.ndarray)
    check(assert_type(arr.astype(Int32Dtype()), ExtensionArray), ExtensionArray)
    check(assert_type(arr.astype("Int64"), ArrayLike), ExtensionArray)

    check(assert_type(arr.fillna(3, limit=1, copy=False), IntegerArray), IntegerArray)
    check(assert_type(arr.fillna(arr), IntegerArray), IntegerArray)

    check(assert_type(arr.view(), IntegerArray), IntegerArray)

    check(assert_type(arr.searchsorted(1), np.intp), np.intp)
    check(assert_type(arr.searchsorted([1]), "np_1darray[np.intp]"), np_1darray)
    check(assert_type(arr.searchsorted(1, side="left"), np.intp), np.intp)
    check(assert_type(arr.searchsorted(1, sorter=[1, 0, 2]), np.intp), np.intp)
