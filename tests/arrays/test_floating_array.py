from collections import UserList
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    PANDAS_FLOAT_ARGS,
    PD_LTE_23,
    check,
    exception_on_platform,
)
from tests._typing import PandasFloatDtypeArg


def test_constructor_sequence() -> None:
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[float | np.float32]", [1.0, np.float32(1)]
    )

    if PD_LTE_23:
        check(assert_type(pd.array([float("nan")]), FloatingArray), NumpyExtensionArray)
    else:
        check(assert_type(pd.array([float("nan")]), FloatingArray), FloatingArray)

    check(assert_type(pd.array(data), FloatingArray), FloatingArray)
    check(assert_type(pd.array([*data, np.nan]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([*data, None]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([*data, pd.NA]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([*data, None, pd.NA]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([*data, np.nan, pd.NA]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([*data, np.nan, None]), FloatingArray), FloatingArray)
    check(
        assert_type(pd.array([*data, np.nan, None, pd.NA]), FloatingArray),
        FloatingArray,
    )

    check(assert_type(pd.array(tuple(data)), FloatingArray), FloatingArray)
    check(assert_type(pd.array(UserList(data)), FloatingArray), FloatingArray)


def test_constructor_array_like() -> None:
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[float | np.float32]", [1.0, np.float32(1)]
    )
    np_arr = np.array(data, np.float32)

    check(assert_type(pd.array(np_arr), FloatingArray), FloatingArray)

    check(assert_type(pd.array(pd.array(data)), FloatingArray), FloatingArray)


@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_FLOAT_ARGS.items(), ids=repr)
def test_constructor_dtype(dtype: PandasFloatDtypeArg, target_dtype: type) -> None:
    exc = exception_on_platform(dtype)
    if exc:
        with pytest.raises(exc, match=rf"data type {dtype!r} not understood"):
            assert_type(pd.array([1.0], dtype=dtype), FloatingArray)
    else:
        check(pd.array([1.0], dtype=dtype), FloatingArray, target_dtype)

    if TYPE_CHECKING:
        # pandas Float32
        assert_type(pd.array([1.0], dtype=pd.Float32Dtype()), FloatingArray)
        assert_type(pd.array([1.0], dtype="Float32"), FloatingArray)
        # pandas Float64
        assert_type(pd.array([1.0], dtype=pd.Float64Dtype()), FloatingArray)
        assert_type(pd.array([1.0], dtype="Float64"), FloatingArray)
