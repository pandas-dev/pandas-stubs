from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.floating import FloatingArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    PD_LTE_23,
    check,
    exception_on_platform,
)
from tests._typing import PandasFloatDtypeArg
from tests.dtypes import PANDAS_FLOAT_ARGS
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", powerset([1.15, np.float32(-2.3)], 1))
@pytest.mark.parametrize("missing_values", powerset([np.nan, None, pd.NA]))
def test_construction_sequence(
    data: tuple[float | np.floating, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    check(pd.array(typ([*data, *missing_values])), FloatingArray)
    check(pd.array(typ([-3.2, *data, *missing_values])), FloatingArray)
    check(pd.array(typ([np.float16(-2e11), *data, *missing_values])), FloatingArray)

    if TYPE_CHECKING:
        assert_type(pd.array([0.1, 1.2]), FloatingArray)
        assert_type(pd.array([np.float64(0.1), np.float64(1.2)]), FloatingArray)

        assert_type(pd.array([1.1, np.float16(2.3)]), FloatingArray)

        assert_type(pd.array([5.8, np.float32(9.8), np.nan]), FloatingArray)
        assert_type(pd.array([7.6, np.float64(5.4), None]), FloatingArray)
        assert_type(pd.array([3.2, -np.float16(0.1), pd.NA]), FloatingArray)

        assert_type(pd.array([-2.3, np.float32(4.5), None, pd.NA]), FloatingArray)
        assert_type(pd.array([6.7, -np.float64(8.9), np.nan, pd.NA]), FloatingArray)
        assert_type(pd.array([-0.1, -np.float16(2.3), np.nan, None]), FloatingArray)

        assert_type(pd.array([4.5, np.float32(0), np.nan, None, pd.NA]), FloatingArray)

        assert_type(pd.array((0.0, np.float64(2.3e64))), FloatingArray)
        assert_type(pd.array((-3.5e64, np.float64(7.2), pd.NA)), FloatingArray)

        assert_type(pd.array(UserList([8.1, np.float32(9.2)])), FloatingArray)


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", [(), (np.nan,)])
def test_construction_sequence_nan(
    data: tuple[Any, ...], typ: Callable[[Sequence[Any]], Sequence[Any]]
) -> None:
    expected_type = (
        NumpyExtensionArray if data == (np.nan,) and PD_LTE_23 else FloatingArray
    )
    check(pd.array(typ(data)), expected_type)

    if TYPE_CHECKING:
        assert_type(pd.array([]), FloatingArray)
        assert_type(pd.array([np.nan]), FloatingArray)

        assert_type(pd.array(()), FloatingArray)
        assert_type(pd.array((np.nan,)), FloatingArray)

        assert_type(pd.array(UserList()), FloatingArray)
        assert_type(pd.array(UserList([np.nan])), FloatingArray)


def test_construction_array_like() -> None:
    np_arr = np.array([1.0, np.float16(1)], np.float32)
    check(assert_type(pd.array(np_arr), FloatingArray), FloatingArray)

    check(
        assert_type(pd.array(pd.array([1.0, np.float32(1)])), FloatingArray),
        FloatingArray,
    )


@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_FLOAT_ARGS.items(), ids=repr)
def test_construction_dtype(dtype: PandasFloatDtypeArg, target_dtype: type) -> None:
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
