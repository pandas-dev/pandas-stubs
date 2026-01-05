"""Test module for methods in pandas.core.arrays.period."""

from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    assert_type,
)

import numpy as np
from pandas import (
    PeriodDtype,
    PeriodIndex,
    Series,
)
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.period import PeriodArray
import pyarrow as pa

from pandas._libs.tslibs.period import Period

from tests import check

if TYPE_CHECKING:
    from pandas._typing import (
        np_1darray_anyint,
        np_1darray_bool,
    )
else:
    np_1darray_anyint = np.ndarray
    np_1darray_bool = np.ndarray


def test_period_array_init() -> None:
    """Test init method for PeriodArray."""
    # From numpy array of integers (ordinals) with dtype
    dtype = PeriodDtype(freq="D")
    values = np.array([1, 2, 3], dtype=np.int64)
    arr = PeriodArray(values, dtype=dtype)
    check(assert_type(arr, PeriodArray), PeriodArray)

    # From PeriodArray
    arr2 = PeriodArray(arr)
    check(assert_type(arr2, PeriodArray), PeriodArray)

    # From PeriodIndex
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr, PeriodArray), PeriodArray)

    # From Series of Period
    periods = Series([Period("2020-01", freq="M"), Period("2020-02", freq="M")])
    arr = PeriodArray(periods)
    check(assert_type(arr, PeriodArray), PeriodArray)

    # With copy parameter
    arr = PeriodArray(idx, copy=True)
    check(assert_type(arr, PeriodArray), PeriodArray)

    arr = PeriodArray(idx, copy=False)
    check(assert_type(arr, PeriodArray), PeriodArray)


def test_period_array_dtype() -> None:
    """Test dtype property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.dtype, PeriodDtype), PeriodDtype)


# def test_period_array_array() -> None:
#     """Test __array__ method for PeriodArray."""
#     idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
#     arr = PeriodArray(idx)
#     check(assert_type(np.asarray(arr), "np_1darray"), np.ndarray)
#     check(assert_type(np.asarray(arr, dtype=object), "np_1darray"), np.ndarray)


def test_period_array_arrow_array() -> None:
    """Test __arrow_array__ method for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(
        assert_type(arr.__arrow_array__(), "pa.ExtensionArray[Any]"), pa.ExtensionArray
    )
    check(
        assert_type(arr.__arrow_array__(type=None), "pa.ExtensionArray[Any]"),
        pa.ExtensionArray,
    )


def test_period_array_year() -> None:
    """Test year property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2021-02", "2022-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.year, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_month() -> None:
    """Test month property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.month, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_day() -> None:
    """Test day property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-01-15", "2020-01-31"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.day, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_hour() -> None:
    """Test hour property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01 10:00", "2020-01-01 14:00"], freq="h")
    arr = PeriodArray(idx)
    check(assert_type(arr.hour, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_minute() -> None:
    """Test minute property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01 10:30", "2020-01-01 10:45"], freq="min")
    arr = PeriodArray(idx)
    check(assert_type(arr.minute, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_second() -> None:
    """Test second property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01 10:30:15", "2020-01-01 10:30:45"], freq="s")
    arr = PeriodArray(idx)
    check(assert_type(arr.second, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_weekofyear() -> None:
    """Test weekofyear property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-06-15", "2020-12-31"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.weekofyear, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_week() -> None:
    """Test week property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-06-15", "2020-12-31"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.week, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_dayofweek() -> None:
    """Test dayofweek property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-01-02", "2020-01-03"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.dayofweek, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_weekday() -> None:
    """Test weekday property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-01-02", "2020-01-03"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.weekday, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_dayofyear() -> None:
    """Test dayofyear property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-06-15", "2020-12-31"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.dayofyear, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_day_of_year() -> None:
    """Test day_of_year property for PeriodArray."""
    idx = PeriodIndex(["2020-01-01", "2020-06-15", "2020-12-31"], freq="D")
    arr = PeriodArray(idx)
    check(assert_type(arr.day_of_year, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_quarter() -> None:
    """Test quarter property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-04", "2020-07", "2020-10"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.quarter, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_qyear() -> None:
    """Test qyear property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-04", "2020-07", "2020-10"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.qyear, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_days_in_month() -> None:
    """Test days_in_month property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.days_in_month, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_daysinmonth() -> None:
    """Test daysinmonth property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.daysinmonth, np_1darray_anyint), np.ndarray, np.integer)


def test_period_array_is_leap_year() -> None:
    """Test is_leap_year property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2021-01", "2024-01"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.is_leap_year, np_1darray_bool), np.ndarray, np.bool_)


def test_period_array_start_time() -> None:
    """Test start_time property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.start_time, DatetimeArray), DatetimeArray)


def test_period_array_end_time() -> None:
    """Test end_time property for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)
    check(assert_type(arr.end_time, DatetimeArray), DatetimeArray)


def test_period_array_to_timestamp() -> None:
    """Test to_timestamp method for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)

    check(assert_type(arr.to_timestamp(), DatetimeArray), DatetimeArray)
    check(assert_type(arr.to_timestamp(freq="D"), DatetimeArray), DatetimeArray)
    check(assert_type(arr.to_timestamp(how="start"), DatetimeArray), DatetimeArray)
    check(assert_type(arr.to_timestamp(how="end"), DatetimeArray), DatetimeArray)
    check(
        assert_type(arr.to_timestamp(freq="D", how="start"), DatetimeArray),
        DatetimeArray,
    )


def test_period_array_asfreq() -> None:
    """Test asfreq method for PeriodArray."""
    idx = PeriodIndex(["2020-01", "2020-02", "2020-03"], freq="M")
    arr = PeriodArray(idx)

    check(assert_type(arr.asfreq(freq="D"), PeriodArray), PeriodArray)
    check(assert_type(arr.asfreq(freq="D", how="start"), PeriodArray), PeriodArray)
    check(assert_type(arr.asfreq(freq="D", how="end"), PeriodArray), PeriodArray)
    check(assert_type(arr.asfreq(freq="D", how="E"), PeriodArray), PeriodArray)
    check(assert_type(arr.asfreq(freq="D", how="S"), PeriodArray), PeriodArray)
