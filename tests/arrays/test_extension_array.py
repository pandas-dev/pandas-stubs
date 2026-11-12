# Test common ExtensionArray methods
from typing import assert_type

import numpy as np
import pandas as pd
from pandas.api.typing.aliases import ArrayLike
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.integer import (
    Int32Dtype,
    IntegerArray,
)
from pandas.core.construction import array

from pandas._libs.missing import NAType

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    np_1darray,
    np_1darray_bool,
    np_1darray_intp,
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
    check(assert_type(arr.count(), int), np.integer)
    check(assert_type(array(["a", None], dtype="string").count(), int), int)
    check(assert_type(arr.copy().sort(), None), type(None))
    check(
        assert_type(
            arr.copy().sort(ascending=False, kind="stable", na_position="first"),
            None,
        ),
        type(None),
    )
    check(assert_type(arr.duplicated(), np_1darray_bool), np_1darray_bool)
    check(assert_type(arr.duplicated(keep=False), np_1darray_bool), np_1darray_bool)
    check(assert_type(arr.item(0), np.integer | NAType), np.int64)
    check(assert_type(array([1]).item(), np.integer | NAType), np.int64)
    check(assert_type(arr.dropna(), IntegerArray), IntegerArray)
    check(assert_type(arr.take([1, 0, 2]), IntegerArray), IntegerArray)
    check(
        assert_type(arr.take([1, 0, 2], allow_fill=True, fill_value=-1), IntegerArray),
        IntegerArray,
    )
    check(assert_type(arr.ravel(), IntegerArray), IntegerArray)

    check(assert_type(arr.astype(np.dtype("int64")), np_1darray), np_1darray)
    check(assert_type(arr.astype(Int32Dtype()), ExtensionArray), ExtensionArray)
    check(assert_type(arr.astype("Int64"), ArrayLike), ExtensionArray)

    check(assert_type(arr.fillna(3, limit=1, copy=False), IntegerArray), IntegerArray)
    check(assert_type(arr.fillna(arr), IntegerArray), IntegerArray)

    check(assert_type(arr.view(), IntegerArray), IntegerArray)

    check(assert_type(arr.searchsorted(1), np.intp), np.intp)
    check(assert_type(arr.searchsorted([1]), np_1darray_intp), np_1darray_intp)
    check(assert_type(arr.searchsorted(1, side="left"), np.intp), np.intp)
    check(assert_type(arr.searchsorted(1, sorter=[1, 0, 2]), np.intp), np.intp)

    # Note: interpolate promotes IntegerArray to FloatingArray, so it is tested
    # here on a nullable float array to keep the result type equal to `Self`.
    float_arr = array([1.0, 2.0, None, 4.0], dtype="Float64")
    check(
        assert_type(
            float_arr.interpolate(
                method="linear",
                axis=0,
                index=pd.Index(range(len(float_arr))),
                limit=None,
                limit_direction="forward",
                limit_area=None,
                copy=True,
            ),
            FloatingArray,
        ),
        FloatingArray,
    )
    if TYPE_CHECKING_INVALID_USAGE:
        arr.copy().sort(False)  # type: ignore[call-arg] # pyright: ignore[reportCallIssue] # pyrefly: ignore[bad-argument-count] # ty: ignore[too-many-positional-arguments]
        arr.copy().sort(kind="foo")  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
        arr.copy().sort(na_position="middle")  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore[bad-argument-type] # ty: ignore[invalid-argument-type]
