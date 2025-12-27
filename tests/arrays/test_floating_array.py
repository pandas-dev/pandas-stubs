from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
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
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("nums", powerset([1.0, np.float32(1)], 1))
@pytest.mark.parametrize("missing_values", powerset([np.nan, None, pd.NA]))
def test_construction_sequence(
    nums: tuple[float | np.floating, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    check(pd.array(typ([*nums, *missing_values])), FloatingArray)

    if TYPE_CHECKING:
        data = cast(  # pyright: ignore[reportUnnecessaryCast]
            "list[float | np.float32]", [1.0, np.float32(1)]
        )

        assert_type(pd.array(data), FloatingArray)
        assert_type(pd.array([*data, np.nan]), FloatingArray)
        assert_type(pd.array([*data, None]), FloatingArray)
        assert_type(pd.array([*data, pd.NA]), FloatingArray)
        assert_type(pd.array([*data, None, pd.NA]), FloatingArray)
        assert_type(pd.array([*data, np.nan, pd.NA]), FloatingArray)
        assert_type(pd.array([*data, np.nan, None]), FloatingArray)
        assert_type(pd.array([*data, np.nan, None, pd.NA]), FloatingArray)

        assert_type(pd.array(tuple(data)), FloatingArray)
        assert_type(pd.array(UserList(data)), FloatingArray)


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", [(), (np.nan,)])
def test_construction_sequence_nan(
    data: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
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
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[float | np.float32]", [1.0, np.float32(1)]
    )
    np_arr = np.array(data, np.float32)

    check(assert_type(pd.array(np_arr), FloatingArray), FloatingArray)

    check(assert_type(pd.array(pd.array(data)), FloatingArray), FloatingArray)


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
