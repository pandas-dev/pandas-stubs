from typing import Union, Optional, Type, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

import pandas as pd

from pandas._typing import Dtype


# The purpose of these checkers is twofold:
# 1) The type checker will see that the right result was returned
# 2) When using pytest, we can check that the result in the pyi file
#    corresponds to what pandas returns


def check_dataframe_result(result: pd.DataFrame):
    assert isinstance(result, pd.DataFrame)


def check_series_result(result: pd.Series, dtype: Optional[Dtype] = None):
    """
    Check that the result is a Series, and that the dtype is the specified dtype

    Since pandas doesn't support typing of a Series, we can check the type
    in pytest instead.

    Parameters
    ----------
    result : pd.Series
        result to check
    dtype : Optional[Dtype], optional
        expected dtype, by default None, which means don't check the dtype
    """
    assert isinstance(result, pd.Series)
    if dtype is not None:
        assert result.dtype == dtype


def check_index_result(result: pd.Index, dtype: Optional[Dtype] = None):
    """
    Check that the result is a Index, and that the dtype is the specified dtype

    Since pandas doesn't support typing of a Index, we can check the type
    in pytest instead.

    Parameters
    ----------
    result : pd.Index
        result to check
    dtype : Optional[Dtype], optional
        expected dtype, by default None, which means don't check the dtype
    """
    assert isinstance(result, pd.Index)
    if dtype is not None:
        assert result.dtype == dtype


def check_multiindex_result(result: pd.MultiIndex):
    """
    Check that the result is a MultiIndex

    Parameters
    ----------
    result : pd.MultiIndex
        A multiindex
    """

    assert isinstance(result, pd.MultiIndex)


def check_datetimeindex_result(result: pd.DatetimeIndex):
    """
    Check that the result is a DatetimeIndex

    Parameters
    ----------
    result : pd.DatetimeIndex
       result to check
    """
    assert isinstance(result, pd.DatetimeIndex)


def check_numpy_result(result: np.ndarray, dtype: Optional[Union[Type[np.int64], Type[np.bool_], Type[np.str_]]] = None):
    assert isinstance(result, np.ndarray)
    if dtype is not None:
        assert result.dtype == dtype


def check_timedelta_result(result: pd.Timedelta):
    assert isinstance(result, pd.Timedelta)


def check_timestamp_result(result: pd.Timestamp):
    assert isinstance(result, pd.Timestamp)


def check_interval_result(
    result: pd.Interval, dtype: Optional[Union[Type[pd.Timestamp], Type[pd.Timedelta], Type[int], Type[float]]]
):
    assert isinstance(result, pd.Interval)
    if dtype is not None:
        assert isinstance(result.left, dtype)


def check_int_result(result: int):
    assert isinstance(result, int)


def check_float_result(result: float):
    assert isinstance(result, float)


def check_bool_result(result: bool):
    assert isinstance(result, bool)
