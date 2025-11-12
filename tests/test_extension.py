import decimal

import numpy as np
import pandas as pd
from pandas.arrays import IntegerArray
from pandas.core.arrays.boolean import BooleanArray
from pandas.core.indexers import check_array_indexer
from typing_extensions import assert_type

from tests import (
    check,
    np_1darray_bool,
)
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
    # python/mypy#19952: mypy believes ExtensionArray and its subclasses have a
    # conflict and gives Any for s.array
    check(assert_type(s.array.tolist(), list), list)  # type: ignore[assert-type]
    check(assert_type(s1.array.tolist(), list), list)
    check(assert_type(pd.array([1, 2, 3]).tolist(), list), list)


def test_ExtensionArray_reduce_accumulate() -> None:
    _data = IntegerArray(
        values=np.array([1, 2, 3], dtype=int),
        mask=np.array([True, False, False], dtype=bool),
    )
    check(assert_type(_data._reduce("max"), object), np.integer)
    check(assert_type(_data._accumulate("cumsum"), IntegerArray), IntegerArray)


def test_array_indexer() -> None:
    arr = pd.array([1, 2])

    m_pd = pd.array([True, False])
    check(assert_type(check_array_indexer(arr, m_pd), np_1darray_bool), np_1darray_bool)

    m_np = np.array([True, False], np.bool_)
    check(assert_type(check_array_indexer(arr, m_np), np_1darray_bool), np_1darray_bool)

    check(assert_type(check_array_indexer(arr, 1), int), int)

    check(assert_type(check_array_indexer(arr, slice(0, 1, 1)), slice), slice)


def test_boolean_array() -> None:
    """Test creation of and operations on BooleanArray GH1411."""
    arr = pd.array([True], dtype="boolean")
    check(assert_type(arr, BooleanArray), BooleanArray)
    arr_and = arr & arr
    check(assert_type(arr_and, BooleanArray), BooleanArray)
