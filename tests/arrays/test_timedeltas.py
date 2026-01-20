"""Test module for methods in pandas.core.arrays.timedeltas."""

from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimelike import DTScalarOrNaT
from pandas.core.arrays.timedeltas import TimedeltaArray
from typing_extensions import assert_type

from pandas._libs import NaTType
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._typing import (
    TimeUnit,
    np_1darray_anyint,
    np_1darray_object,
)

from tests import check
from tests._typing import (
    np_1darray_float,
    np_1darray_int32,
    np_1darray_td,
)


def test_construction() -> None:
    """Test pd.array method for TimedeltaArray."""
    # From TimedeltaIndex
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)
    check(assert_type(arr, TimedeltaArray), TimedeltaArray)

    # From numpy array of timedelta64
    values = np.array(
        [np.timedelta64(1, "D"), np.timedelta64(2, "D"), np.timedelta64(3, "D")]
    )
    arr = pd.array(cast(np_1darray_td, values))
    check(assert_type(arr, TimedeltaArray), TimedeltaArray)

    # With dtype parameter
    arr = pd.array(idx, dtype="timedelta64[ns]")
    check(assert_type(arr, TimedeltaArray), TimedeltaArray)

    # With copy parameter
    arr = pd.array(idx, copy=True)
    check(assert_type(arr, TimedeltaArray), TimedeltaArray)

    arr = pd.array(idx, copy=False)
    check(assert_type(arr, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_dtype() -> None:
    """Test dtype property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)
    check(
        assert_type(arr.dtype, np.dtypes.TimeDelta64DType), np.dtypes.TimeDelta64DType
    )


def test_timedelta_array_mul() -> None:
    """Test __mul__ and __rmul__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr * 2
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = 2 * arr
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr * 0.5
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_truediv() -> None:
    """Test __truediv__ and __rtruediv__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr / 2
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result_np = arr / Timedelta("1 day")
    check(assert_type(result_np, np_1darray_float), np_1darray_float)


def test_timedelta_array_floordiv() -> None:
    """Test __floordiv__ and __rfloordiv__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr // 2
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result_np = arr // Timedelta("1 day")
    check(assert_type(result_np, np_1darray_float), np_1darray_float)


def test_timedelta_array_mod() -> None:
    """Test __mod__ and __rmod__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr % Timedelta("12 hours")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_divmod() -> None:
    """Test __divmod__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = divmod(arr, Timedelta("12 hours"))
    q, r = check(assert_type(result, tuple[np_1darray_anyint, TimedeltaArray]), tuple)
    check(assert_type(q, np_1darray_anyint), np_1darray_anyint)
    check(assert_type(r, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_neg() -> None:
    """Test __neg__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = -arr
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_pos() -> None:
    """Test __pos__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = +arr
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_abs() -> None:
    """Test __abs__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["-1 days", "2 days", "-3 days"])
    arr = pd.array(idx)

    result = abs(arr)
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_total_seconds() -> None:
    """Test total_seconds method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.total_seconds()
    check(assert_type(result, np_1darray_float), np_1darray_float)


def test_timedelta_array_to_pytimedelta() -> None:
    """Test to_pytimedelta method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.to_pytimedelta()
    check(assert_type(result, np_1darray_object), np_1darray_object, timedelta)


def test_timedelta_array_days() -> None:
    """Test days property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 12:00:00", "2 days 06:00:00", "3 days"])
    arr = pd.array(idx)

    result = arr.days
    check(assert_type(result, np_1darray_int32), np_1darray_int32)


def test_timedelta_array_seconds() -> None:
    """Test seconds property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 12:00:00", "2 days 06:00:00", "3 days"])
    arr = pd.array(idx)

    result = arr.seconds
    check(assert_type(result, np_1darray_int32), np_1darray_int32)


def test_timedelta_array_microseconds() -> None:
    """Test microseconds property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 00:00:00.000100", "2 days 00:00:00.000200"])
    arr = pd.array(idx)

    result = arr.microseconds
    check(assert_type(result, np_1darray_int32), np_1darray_int32)


def test_timedelta_array_nanoseconds() -> None:
    """Test nanoseconds property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 00:00:00.000000100", "2 days 00:00:00.000000200"])
    arr = pd.array(idx)

    result = arr.nanoseconds
    check(assert_type(result, np_1darray_int32), np_1darray_int32)


def test_timedelta_array_components() -> None:
    """Test components property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 12:30:45.123456", "2 days 06:15:30.654321"])
    arr = pd.array(idx)

    result = arr.components
    check(assert_type(result, pd.DataFrame), pd.DataFrame)


# Tests for inherited methods from DatetimeLikeArrayMixin and TimelikeOps


def test_timedelta_array_unit() -> None:
    """Test unit property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.unit
    check(assert_type(result, TimeUnit), str)


def test_timedelta_array_as_unit() -> None:
    """Test as_unit method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.as_unit("s")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr.as_unit("ms")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr.as_unit("us")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr.as_unit("ns")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_round() -> None:
    """Test round method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 12:30:45", "2 days 06:15:30"])
    arr = pd.array(idx)

    result = arr.round("h")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr.round("D")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_floor() -> None:
    """Test floor method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 12:30:45", "2 days 06:15:30"])
    arr = pd.array(idx)

    result = arr.floor("h")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr.floor("D")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_ceil() -> None:
    """Test ceil method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days 12:30:45", "2 days 06:15:30"])
    arr = pd.array(idx)

    result = arr.ceil("h")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)

    result = arr.ceil("D")
    check(assert_type(result, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_ndim() -> None:
    """Test ndim property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.ndim
    check(assert_type(result, int), int)


def test_timedelta_array_nbytes() -> None:
    """Test nbytes property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.nbytes
    check(assert_type(result, int), int)


def test_timedelta_array_size() -> None:
    """Test size property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.size
    check(assert_type(result, int), int)


def test_timedelta_array_freq() -> None:
    """Test freq property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")
    arr = pd.array(idx)

    result = arr.freq
    check(assert_type(result, pd.DateOffset | None), pd.DateOffset)


def test_timedelta_array_freqstr() -> None:
    """Test freqstr property for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"], freq="D")
    arr = pd.array(idx)

    result = arr.freqstr
    check(assert_type(result, str | None), str)


def test_timedelta_array_min() -> None:
    """Test min method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.min()
    check(assert_type(result, Timedelta | NaTType), Timedelta)

    result = arr.min(skipna=True)
    check(assert_type(result, Timedelta | NaTType), Timedelta)


def test_timedelta_array_max() -> None:
    """Test max method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.max()
    check(assert_type(result, Timedelta | NaTType), Timedelta)

    result = arr.max(skipna=True)
    check(assert_type(result, Timedelta | NaTType), Timedelta)


def test_timedelta_array_mean() -> None:
    """Test mean method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.mean()
    check(assert_type(result, Timedelta | NaTType), Timedelta)

    result = arr.mean(skipna=True)
    check(assert_type(result, Timedelta | NaTType), Timedelta)


def test_timedelta_array_getitem() -> None:
    """Test __getitem__ for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    # Scalar indexing
    result = arr[0]
    check(assert_type(result, DTScalarOrNaT), Timedelta)

    # Slice indexing
    result_slice = arr[0:2]
    check(assert_type(result_slice, TimedeltaArray), TimedeltaArray)

    # List indexing
    result_lst = arr[[0, 2]]
    check(assert_type(result_lst, TimedeltaArray), TimedeltaArray)

    # Boolean indexing
    result_arr = arr[np.array([True, False, True])]
    check(assert_type(result_arr, TimedeltaArray), TimedeltaArray)


def test_timedelta_array_array() -> None:
    """Test __array__ method for TimedeltaArray."""
    idx = pd.TimedeltaIndex(["1 days", "2 days", "3 days"])
    arr = pd.array(idx)

    result = arr.__array__()
    check(assert_type(result, np_1darray_td), np_1darray_td)
