import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.typing import NAType
from typing_extensions import assert_type

from tests import check


def test_arithmetic() -> None:
    na = pd.NA
    s_int = pd.Series([1, 2, 3])
    idx_int = pd.Index([1, 2, 3])
    arr_int = np.array([1, 2, 3])

    # __add__
    check(assert_type(na + s_int, pd.Series), pd.Series)
    check(assert_type(na + idx_int, pd.Index), pd.Index)
    check(assert_type(na + arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na + 1, NAType), NAType)

    # __sub__
    check(assert_type(na - s_int, pd.Series), pd.Series)
    check(assert_type(na - idx_int, pd.Index), pd.Index)
    check(assert_type(na - arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na - 1, NAType), NAType)

    # __mul__
    check(assert_type(na * s_int, pd.Series), pd.Series)
    check(assert_type(na * idx_int, pd.Index), pd.Index)
    check(assert_type(na * arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na * 1, NAType), NAType)

    # __floordiv__
    check(assert_type(na // s_int, pd.Series), pd.Series)
    check(assert_type(na // idx_int, pd.Index), pd.Index)
    check(assert_type(na // arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na // 1, NAType), NAType)

    # __truediv__
    check(assert_type(na / s_int, pd.Series), pd.Series)
    check(assert_type(na / idx_int, pd.Index), pd.Index)
    check(assert_type(na / arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na / 1, NAType), NAType)

    # __mod__
    check(assert_type(na % s_int, pd.Series), pd.Series)
    check(assert_type(na % idx_int, pd.Index), pd.Index)
    check(assert_type(na % arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na % 1, NAType), NAType)

    # __eq__
    check(assert_type(na == s_int, pd.Series), pd.Series)
    check(assert_type(na == idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na == arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na == 1, NAType), NAType)

    # __ne__
    check(assert_type(na != s_int, pd.Series), pd.Series)
    check(assert_type(na != idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na != arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na != 1, NAType), NAType)

    # __le__
    check(assert_type(na <= s_int, pd.Series), pd.Series)
    check(assert_type(na <= idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na <= arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na <= 1, NAType), NAType)

    # __lt__
    check(assert_type(na < s_int, pd.Series), pd.Series)
    check(assert_type(na < idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na < arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na < 1, NAType), NAType)

    # __ge__
    check(assert_type(na >= s_int, pd.Series), pd.Series)
    check(assert_type(na >= idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na >= arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na >= 1, NAType), NAType)

    # __gt__
    check(assert_type(na > s_int, pd.Series), pd.Series)
    check(assert_type(na > idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na > arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na > 1, NAType), NAType)

    # __pow__
    check(assert_type(na**s_int, pd.Series), pd.Series)
    check(assert_type(na**idx_int, pd.Index), pd.Index)
    check(assert_type(na**arr_int, npt.NDArray), npt.NDArray)
    check(assert_type(na**2, NAType), NAType)

    # __and__
    check(assert_type(na & s_int, pd.Series), pd.Series)
    check(assert_type(na & idx_int, pd.Index), pd.Index)
    check(assert_type(na & arr_int, npt.NDArray), npt.NDArray)
    # check(assert_type(na & True, NAType), NAType)

    # __or__
    check(assert_type(na | s_int, pd.Series), pd.Series)
    check(assert_type(na | idx_int, pd.Index), pd.Index)
    check(assert_type(na | arr_int, npt.NDArray), npt.NDArray)
    # check(assert_type(na | True, NAType), NAType)

    # __xor__
    check(assert_type(na ^ s_int, pd.Series), pd.Series)
    check(assert_type(na ^ idx_int, pd.Index), pd.Index)
    check(assert_type(na ^ arr_int, npt.NDArray), npt.NDArray)
