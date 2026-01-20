"""Test module for methods in pandas.core.arrays.sparse.array."""

from typing import (
    Any,
    Literal,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray
from typing_extensions import assert_type

from pandas._libs.sparse import SparseIndex
from pandas._typing import Scalar

from pandas.core.dtypes.dtypes import SparseDtype

from tests import (
    PD_LTE_23,
    check,
)
from tests._typing import (
    np_1darray,
    np_1darray_int32,
    np_ndarray,
)


def test_constructor() -> None:
    """Test __new__ method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray(np.array([1, 0, 0, 2, 3]))
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray(pd.Series([1, 0, 0, 2, 3]))
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1, 0, 0, 2, 3], kind="integer")
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1, 0, 0, 2, 3], kind="block")
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1.0, 0.0, 0.0, 2.0, 3.0], dtype=np.float64)
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1, 0, 0, 2, 3], dtype=np.int64)
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1, 0, 0, 2, 3], dtype=SparseDtype(dtype=int, fill_value=0))
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([1, 0, 0, 2, 3], copy=True)
    check(assert_type(arr, SparseArray), SparseArray)

    arr = SparseArray([])
    check(assert_type(arr, SparseArray), SparseArray)


def test_sparse_dtype() -> None:
    """Test dtype property for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])
    check(assert_type(arr.dtype, SparseDtype), SparseDtype)


def test_sparse_properties() -> None:
    """Test properties for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)

    check(assert_type(arr.sp_index, SparseIndex), SparseIndex)
    check(assert_type(arr.sp_values, np_ndarray), np.ndarray)
    check(assert_type(arr.dtype, SparseDtype), SparseDtype)
    check(assert_type(arr.nbytes, int), int)
    check(assert_type(arr.density, float), float)
    check(assert_type(arr.npoints, int), int)


def test_sparse_fill_value() -> None:
    """Test fill_value property for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)
    check(assert_type(arr.fill_value, Any), int)


def test_sparse_kind() -> None:
    """Test kind property for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], kind="integer")
    assert assert_type(arr.kind, Literal["integer", "block"]) in {"integer", "block"}


def test_sparse_shift() -> None:
    """Test shift method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)

    check(assert_type(arr.shift(), SparseArray), SparseArray)
    check(assert_type(arr.shift(periods=2), SparseArray), SparseArray)
    check(assert_type(arr.shift(periods=-1), SparseArray), SparseArray)
    check(assert_type(arr.shift(periods=1, fill_value=0), SparseArray), SparseArray)


def test_sparse_unique() -> None:
    """Test unique method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3, 1])
    check(assert_type(arr.unique(), SparseArray), SparseArray)


def test_sparse_value_counts() -> None:
    """Test value_counts method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3, 1])
    check(assert_type(arr.value_counts(), "pd.Series[int]"), pd.Series, np.integer)
    check(
        assert_type(arr.value_counts(dropna=True), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )
    check(
        assert_type(arr.value_counts(dropna=False), "pd.Series[int]"),
        pd.Series,
        np.integer,
    )


def test_sparse_getitem() -> None:
    """Test __getitem__ method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])

    # Scalar indexer
    check(assert_type(arr[0], Any), np.integer)
    check(assert_type(arr[-1], Any), np.integer)

    # Sequence indexer
    check(assert_type(arr[[0, 1, 2]], SparseArray), SparseArray)
    check(assert_type(arr[np.array([0, 1, 2])], SparseArray), SparseArray)
    check(assert_type(arr[1:3], SparseArray), SparseArray)


def test_sparse_copy() -> None:
    """Test copy method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])
    check(assert_type(arr.copy(), SparseArray), SparseArray)


def test_sparse_to_dense() -> None:
    """Test to_dense method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])
    check(assert_type(arr.to_dense(), np_1darray), np_1darray, np.integer)


def test_sparse_nonzero() -> None:
    """Test nonzero method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)
    result = arr.nonzero()
    (res,) = check(assert_type(result, tuple[np_1darray_int32]), tuple)
    check(assert_type(res, np_1darray_int32), np_1darray_int32)


def test_sparse_all() -> None:
    """Test all method for SparseArray."""
    arr = SparseArray([1, 1, 1, 1, 1])
    check(assert_type(arr.all(), bool), np.bool_)
    check(assert_type(arr.all(axis=None), bool), np.bool_)
    check(assert_type(arr.all(axis=0), bool), np.bool_)


def test_sparse_any() -> None:
    """Test any method for SparseArray."""
    arr = SparseArray([0, 0, 1, 0, 0], fill_value=0)
    check(assert_type(arr.any(), bool), bool)
    check(assert_type(arr.any(axis=0), bool), bool)


def test_sparse_sum() -> None:
    """Test sum method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)
    check(assert_type(arr.sum(), Scalar), np.integer)
    check(assert_type(arr.sum(axis=0), Scalar), np.integer)
    check(assert_type(arr.sum(min_count=1), Scalar), np.integer)
    check(assert_type(arr.sum(skipna=True), Scalar), np.integer)


def test_sparse_cumsum() -> None:
    """
    Test cumsum method for SparseArray.

    Note: At runtime, cumsum has a recursion bug in pandas.
    This test only validates the type signature.
    """
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)
    if not PD_LTE_23:
        # TODO: pandas-dev/pandas#62669 fix is in 3.0
        check(assert_type(arr.cumsum(), SparseArray), SparseArray)
        check(assert_type(arr.cumsum(axis=0), SparseArray), SparseArray)


def test_sparse_mean() -> None:
    """Test mean method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3], fill_value=0)
    check(assert_type(arr.mean(), SparseArray), float)
    check(assert_type(arr.mean(axis=0), SparseArray), float)


def test_sparse_transpose() -> None:
    """Test T property for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])
    check(assert_type(arr.T, SparseArray), SparseArray)


def test_sparse_abs() -> None:
    """Test __abs__ method for SparseArray."""
    arr = SparseArray([-1, 0, 0, -2, 3])
    check(assert_type(abs(arr), SparseArray), SparseArray)


def test_sparse_array() -> None:
    """Test __array__ method for SparseArray."""
    arr = SparseArray([1, 0, 0, 2, 3])
    check(assert_type(arr.__array__(), np_1darray), np.ndarray)
    check(
        assert_type(arr.__array__(dtype=np.float64), np_1darray), np_1darray, np.float64
    )
