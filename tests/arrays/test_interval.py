"""Test module for methods in pandas.core.arrays.interval."""

import numpy as np
import pandas as pd
from pandas import (
    Index,
    Interval,
    Series,
)
from pandas.core.arrays import IntervalArray
from typing_extensions import assert_type

from pandas.core.dtypes.dtypes import IntervalDtype

from tests import check
from tests._typing import (
    np_1darray_bool,
    np_1darray_object,
)


def test_constructor() -> None:
    """Test __new__ method for IntervalArray."""
    intervals = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
    arr = IntervalArray(intervals)
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, closed="right")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, closed="left")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, closed="both")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, closed="neither")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, copy=True)
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, verify_integrity=True)
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray(intervals, verify_integrity=False)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_constructor_from_index() -> None:
    """Test __new__ method for IntervalArray from Index."""
    idx = pd.IntervalIndex.from_breaks([0, 1, 2, 3])
    arr = IntervalArray(idx)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_constructor_from_series() -> None:
    """Test __new__ method for IntervalArray from Series."""
    intervals = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
    series = Series(intervals)
    arr = IntervalArray(series)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_breaks() -> None:
    """Test from_breaks class method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_breaks([0, 1, 2, 3], closed="right")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_breaks([0, 1, 2, 3], closed="left")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_breaks([0, 1, 2, 3], copy=True)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_breaks_ndarray() -> None:
    """Test from_breaks class method for IntervalArray with ndarray."""
    breaks = np.array([0, 1, 2, 3])
    arr = IntervalArray.from_breaks(breaks)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_breaks_index() -> None:
    """Test from_breaks class method for IntervalArray with Index."""
    breaks = Index([0, 1, 2, 3])
    arr = IntervalArray.from_breaks(breaks)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_breaks_series() -> None:
    """Test from_breaks class method for IntervalArray with Series."""
    breaks = Series([0, 1, 2, 3])
    arr = IntervalArray.from_breaks(breaks)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_arrays() -> None:
    """Test from_arrays class method for IntervalArray."""
    arr = IntervalArray.from_arrays([0, 1, 2], [1, 2, 3])
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_arrays([0, 1, 2], [1, 2, 3], closed="right")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_arrays([0, 1, 2], [1, 2, 3], closed="left")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_arrays([0, 1, 2], [1, 2, 3], copy=True)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_arrays_ndarray() -> None:
    """Test from_arrays class method for IntervalArray with ndarray."""
    left = np.array([0, 1, 2])
    right = np.array([1, 2, 3])
    arr = IntervalArray.from_arrays(left, right)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_arrays_index() -> None:
    """Test from_arrays class method for IntervalArray with Index."""
    left = Index([0, 1, 2])
    right = Index([1, 2, 3])
    arr = IntervalArray.from_arrays(left, right)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_arrays_series() -> None:
    """Test from_arrays class method for IntervalArray with Series."""
    left = Series([0, 1, 2])
    right = Series([1, 2, 3])
    arr = IntervalArray.from_arrays(left, right)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_from_tuples() -> None:
    """Test from_tuples class method for IntervalArray."""
    arr = IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)])
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)], closed="right")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)], closed="left")
    check(assert_type(arr, IntervalArray), IntervalArray)

    arr = IntervalArray.from_tuples([(0, 1), (1, 2), (2, 3)], copy=True)
    check(assert_type(arr, IntervalArray), IntervalArray)


def test_array() -> None:
    """Test __array__ method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr.__array__()
    check(assert_type(result, np_1darray_object), np_1darray_object)


def test_getitem_scalar() -> None:
    """Test __getitem__ with scalar indexer for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr[0]
    check(assert_type(result, "Interval | float"), Interval)


def test_getitem_sequence() -> None:
    """Test __getitem__ with sequence indexer for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr[[0, 1]]
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr[np.array([0, 1])]
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr[0:2]
    check(assert_type(result, IntervalArray), IntervalArray)


def test_eq_ne() -> None:
    """Test __eq__ and __ne__ methods for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    other = Interval(0, 1)

    eq_result = arr == other
    check(assert_type(eq_result, np_1darray_bool), np_1darray_bool)

    ne_result = arr != other
    check(assert_type(ne_result, np_1darray_bool), np_1darray_bool)


def test_properties() -> None:
    """Test properties for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])

    check(assert_type(arr.dtype, IntervalDtype), IntervalDtype)
    check(assert_type(arr.nbytes, int), int)
    check(assert_type(arr.size, int), int)


def test_left_right() -> None:
    """Test left and right properties for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])

    check(assert_type(arr.left, Index), Index)
    check(assert_type(arr.right, Index), Index)


def test_closed() -> None:
    """Test closed property for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    check(assert_type(arr.closed, bool), str)


def test_set_closed() -> None:
    """Test set_closed method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])

    result = arr.set_closed("left")
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr.set_closed("right")
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr.set_closed("both")
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr.set_closed("neither")
    check(assert_type(result, IntervalArray), IntervalArray)


def test_length() -> None:
    """Test length property for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    check(assert_type(arr.length, Index), Index)


def test_mid() -> None:
    """Test mid property for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    check(assert_type(arr.mid, Index), Index)


def test_is_non_overlapping_monotonic() -> None:
    """Test is_non_overlapping_monotonic property for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    check(assert_type(arr.is_non_overlapping_monotonic, bool), bool)


def test_shift() -> None:
    """Test shift method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr.shift()
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr.shift(periods=2)
    check(assert_type(result, IntervalArray), IntervalArray)


def test_take() -> None:
    """Test take method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr.take([0, 1])
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr.take(np.array([0, 1]))
    check(assert_type(result, IntervalArray), IntervalArray)

    result = arr.take([0, 1], allow_fill=False)
    check(assert_type(result, IntervalArray), IntervalArray)


def test_to_tuples() -> None:
    """Test to_tuples method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr.to_tuples()
    check(assert_type(result, np_1darray_object), np_1darray_object)

    result = arr.to_tuples(na_tuple=True)
    check(assert_type(result, np_1darray_object), np_1darray_object)

    result = arr.to_tuples(na_tuple=False)
    check(assert_type(result, np_1darray_object), np_1darray_object)


def test_contains_scalar() -> None:
    """Test contains method with scalar for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    result = arr.contains(0.5)
    check(assert_type(result, np_1darray_bool), np_1darray_bool)


def test_contains_index() -> None:
    """Test contains method with Index for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    idx = Index([0.5, 1.5, 2.5])
    result = arr.contains(idx)
    check(assert_type(result, np_1darray_bool), np_1darray_bool)


def test_contains_ndarray() -> None:
    """Test contains method with ndarray for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    values = np.array([0.5, 1.5, 2.5])
    result = arr.contains(values)
    check(assert_type(result, np_1darray_bool), np_1darray_bool)


def test_contains_series() -> None:
    """Test contains method with Series for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    series = Series([0.5, 1.5, 2.5])
    result = arr.contains(series)
    check(assert_type(result, "Series[bool]"), Series, np.bool_)


def test_overlaps() -> None:
    """Test overlaps method for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])
    other = Interval(0.5, 1.5)
    result = arr.overlaps(other)
    check(assert_type(result, np_1darray_bool), np_1darray_bool)


def test_interval_mixin_properties() -> None:
    """Test IntervalMixin properties for IntervalArray."""
    arr = IntervalArray.from_breaks([0, 1, 2, 3])

    check(assert_type(arr.closed_left, bool), bool)
    check(assert_type(arr.closed_right, bool), bool)
    check(assert_type(arr.open_left, bool), bool)
    check(assert_type(arr.open_right, bool), bool)
    check(assert_type(arr.is_empty, np_1darray_bool), np_1darray_bool)
