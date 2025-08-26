from typing import Any

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
    arr_int: npt.NDArray[Any] = np.array([1, 2, 3])
    ma_int: npt.NDArray[Any] = np.array([[1, 2, 3], [4, 5, 6]])

    # __add__
    check(assert_type(na + s_int, pd.Series), pd.Series)
    check(assert_type(na + idx_int, pd.Index), pd.Index)
    check(assert_type(na + arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na + 1, NAType), NAType)

    # __radd__
    check(assert_type(s_int + na, pd.Series), pd.Series)
    # https://github.com/pandas-dev/pandas-stubs/issues/1347
    check(assert_type(idx_int + na, pd.Index), pd.Index)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(arr_int + na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 + na, NAType), NAType)

    # __sub__
    check(assert_type(na - s_int, pd.Series), pd.Series)
    check(assert_type(na - idx_int, pd.Index), pd.Index)
    check(assert_type(na - arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na - 1, NAType), NAType)

    # __rsub__
    check(assert_type(s_int - na, pd.Series), pd.Series)
    check(assert_type(idx_int - na, pd.Index), pd.Index)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(arr_int - na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 - na, NAType), NAType)

    # __mul__
    check(assert_type(na * s_int, pd.Series), pd.Series)
    check(assert_type(na * idx_int, pd.Index), pd.Index)
    check(assert_type(na * arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na * 1, NAType), NAType)

    # __rmul__
    check(assert_type(s_int * na, pd.Series), pd.Series)
    check(assert_type(idx_int * na, pd.Index), pd.Index)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(arr_int * na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 * na, NAType), NAType)

    # __matmul__
    check(assert_type(na @ ma_int, npt.NDArray), npt.NDArray)
    check(assert_type(na @ 1, NAType), NAType)

    # __rmatmul__
    check(assert_type(1 @ na, NAType), NAType)

    # __truediv__
    check(assert_type(na / s_int, pd.Series), pd.Series)
    check(assert_type(na / idx_int, pd.Index), pd.Index)
    check(assert_type(na / arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na / 1, NAType), NAType)

    # __rtruediv__
    check(assert_type(s_int / na, pd.Series), pd.Series)
    check(assert_type(idx_int / na, pd.Index), pd.Index)
    check(assert_type(arr_int / na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 / na, NAType), NAType)

    # __floordiv__
    check(assert_type(na // s_int, pd.Series), pd.Series)
    check(assert_type(na // idx_int, pd.Index), pd.Index)
    check(assert_type(na // arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na // 1, NAType), NAType)

    # __rfloordiv__
    check(assert_type(s_int // na, pd.Series), pd.Series)
    check(assert_type(idx_int // na, npt.NDArray), npt.NDArray)
    check(assert_type(arr_int // na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 // na, NAType), NAType)

    # __mod__
    check(assert_type(na % s_int, pd.Series), pd.Series)
    check(assert_type(na % idx_int, pd.Index), pd.Index)
    check(assert_type(na % arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na % 1, NAType), NAType)

    # __rmod__
    check(assert_type(s_int % na, pd.Series), pd.Series)
    check(assert_type(idx_int % na, pd.Index), pd.Index)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(arr_int % na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 % na, NAType), NAType)

    # __eq__
    check(assert_type(na == s_int, pd.Series), pd.Series)
    check(assert_type(na == idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na == arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na == 1, NAType), NAType)

    # __req__
    # check(assert_type(= s_int=na, pd.Series), pd.Series)
    # check(assert_type(= idx_int=na, npt.NDArray), npt.NDArray)
    # check(assert_type(= arr_int=na, npt.NDArray), npt.NDArray)
    # check(assert_type(= 1=na, NAType), NAType)

    # __ne__
    check(assert_type(na != s_int, pd.Series), pd.Series)
    check(assert_type(na != idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na != arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na != 1, NAType), NAType)

    # __rne__
    # check(assert_type(= s_int!na, pd.Series), pd.Series)
    # check(assert_type(= idx_int!na, npt.NDArray), npt.NDArray)
    # check(assert_type(= arr_int!na, npt.NDArray), npt.NDArray)
    # check(assert_type(= 1!na, NAType), NAType)

    # __le__
    check(assert_type(na <= s_int, pd.Series), pd.Series)
    check(assert_type(na <= idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na <= arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na <= 1, NAType), NAType)

    # __rle__
    # check(assert_type(= s_int<na, pd.Series), pd.Series)
    # check(assert_type(= idx_int<na, npt.NDArray), npt.NDArray)
    # check(assert_type(= arr_int<na, npt.NDArray), npt.NDArray)
    # check(assert_type(= 1<na, NAType), NAType)

    # __lt__
    check(assert_type(na < s_int, pd.Series), pd.Series)
    check(assert_type(na < idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na < arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na < 1, NAType), NAType)

    # __rlt__
    check(assert_type(s_int < na, pd.Series), pd.Series)
    check(assert_type(idx_int < na, npt.NDArray), npt.NDArray)
    check(assert_type(arr_int < na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 < na, NAType), NAType)

    # __ge__
    check(assert_type(na >= s_int, pd.Series), pd.Series)
    check(assert_type(na >= idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na >= arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na >= 1, NAType), NAType)

    # __rge__
    # check(assert_type(= s_int>na, pd.Series), pd.Series)
    # check(assert_type(= idx_int>na, npt.NDArray), npt.NDArray)
    # check(assert_type(= arr_int>na, npt.NDArray), npt.NDArray)
    # check(assert_type(= 1>na, NAType), NAType)

    # __gt__
    check(assert_type(na > s_int, pd.Series), pd.Series)
    check(assert_type(na > idx_int, npt.NDArray), npt.NDArray)
    check(assert_type(na > arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na > 1, NAType), NAType)

    # __rgt__
    check(assert_type(s_int > na, pd.Series), pd.Series)
    check(assert_type(idx_int > na, npt.NDArray), npt.NDArray)
    check(assert_type(arr_int > na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(1 > na, NAType), NAType)

    # __pow__
    check(assert_type(na**s_int, pd.Series), pd.Series)
    check(assert_type(na**idx_int, pd.Index), pd.Index)
    check(assert_type(na**arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    check(assert_type(na**2, NAType), NAType)

    # __rpow__
    check(assert_type(s_int * na, pd.Series), pd.Series)
    check(assert_type(idx_int * na, pd.Index), pd.Index)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(arr_int * na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    check(assert_type(2 * na, NAType), NAType)

    # __and__
    check(assert_type(na & s_int, pd.Series), pd.Series)
    check(assert_type(na & idx_int, pd.Index), pd.Index)
    check(assert_type(na & arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    # check(assert_type(na & True, NAType), NAType)

    # __rand__
    check(assert_type(s_int & na, pd.Series), pd.Series)
    check(assert_type(idx_int & na, pd.Index), pd.Index)
    check(assert_type(arr_int & na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    # check(ssert_typa & Trueana, NAType), NAType)

    # __or__
    check(assert_type(na | s_int, pd.Series), pd.Series)
    check(assert_type(na | idx_int, pd.Index), pd.Index)
    check(assert_type(na | arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine
    # check(assert_type(na | True, NAType), NAType)

    # __ror__
    check(assert_type(s_int | na, pd.Series), pd.Series)
    check(assert_type(idx_int | na, pd.Index), pd.Index)
    check(assert_type(arr_int | na, npt.NDArray), npt.NDArray)  # type: ignore[assert-type]  # pyright: ignore[reportAssertTypeFailure]
    # check(ssert_typa | Trueana, NAType), NAType)

    # __xor__
    check(assert_type(na ^ s_int, pd.Series), pd.Series)
    check(assert_type(na ^ idx_int, pd.Index), pd.Index)
    check(assert_type(na ^ arr_int, npt.NDArray), npt.NDArray)  # type: ignore[assert-type] # mypy bug? pyright fine

    # rxor
