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
from pandas.core.arrays.integer import (
    IntegerArray,
    IntegerDtype,
)
import pytest
from typing_extensions import assert_type

from pandas._libs.missing import NAType

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import (
    PandasIntDtypeArg,
    PandasUIntDtypeArg,
)
from tests.dtypes import (
    PANDAS_INT_ARGS,
    PANDAS_UINT_ARGS,
)
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", powerset([1, np.int8(1)], 1))
@pytest.mark.parametrize("missing_values", powerset([np.nan, None, pd.NA]))
def test_construction_sequence(
    data: tuple[int | np.integer, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    check(pd.array(typ([*data, *missing_values])), IntegerArray)

    if TYPE_CHECKING:
        assert_type(pd.array([-2, 3]), IntegerArray)
        assert_type(pd.array([1 << 32, np.int8(1) << 6]), IntegerArray)

        assert_type(pd.array([2, np.int8(3)]), IntegerArray)

        assert_type(pd.array([5, np.int16(0o10), None]), IntegerArray)
        assert_type(pd.array([0xD, np.int16(21), pd.NA]), IntegerArray)

        assert_type(pd.array([34, np.int32(55), None, pd.NA]), IntegerArray)

        assert_type(pd.array((8_9, np.int32(144))), IntegerArray)
        assert_type(pd.array((233, np.int32(377), pd.NA)), IntegerArray)

        assert_type(pd.array(UserList([610, np.int64(987)])), IntegerArray)


def test_construction_array_like() -> None:
    np_arr = np.array([1, np.int8(1)], np.int32)
    check(assert_type(pd.array(np_arr), IntegerArray), IntegerArray)

    check(assert_type(pd.array(pd.array([1, np.int16(1)])), IntegerArray), IntegerArray)


@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_INT_ARGS.items(), ids=repr)
def test_construction_dtype_signed(
    dtype: PandasIntDtypeArg, target_dtype: type
) -> None:
    check(pd.array([np.nan], dtype), IntegerArray, NAType)
    check(pd.array([-1, 2, np.nan], dtype), IntegerArray, target_dtype)
    check(pd.array([1, -np.int64(2), np.nan], dtype), IntegerArray, target_dtype)

    if TYPE_CHECKING:
        # pandas Int8
        assert_type(pd.array([-1, 2, np.nan], pd.Int8Dtype()), IntegerArray)
        assert_type(pd.array([-1, 2, np.nan], "Int8"), IntegerArray)
        # pandas Int16
        assert_type(pd.array([-1, 2, np.nan], pd.Int16Dtype()), IntegerArray)
        assert_type(pd.array([-1, 2, np.nan], "Int16"), IntegerArray)
        # pandas Int32
        assert_type(pd.array([-1, 2, np.nan], pd.Int32Dtype()), IntegerArray)
        assert_type(pd.array([-1, 2, np.nan], "Int32"), IntegerArray)
        # pandas Int64
        assert_type(pd.array([-1, 2, np.nan], pd.Int64Dtype()), IntegerArray)
        assert_type(pd.array([-1, 2, np.nan], "Int64"), IntegerArray)


@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_UINT_ARGS.items(), ids=repr)
def test_construction_dtype_unsigned(
    dtype: PandasUIntDtypeArg, target_dtype: type
) -> None:
    check(pd.array([np.nan], dtype), IntegerArray, NAType)
    check(pd.array([1, 2], dtype), IntegerArray, target_dtype)
    check(pd.array([1, np.uint64(2), np.nan], dtype), IntegerArray, target_dtype)

    if TYPE_CHECKING:
        # pandas UInt8
        assert_type(pd.array([1, 2, np.nan], pd.UInt8Dtype()), IntegerArray)
        assert_type(pd.array([1, 2, np.nan], "UInt8"), IntegerArray)
        # pandas UInt16
        assert_type(pd.array([1, 2, np.nan], pd.UInt16Dtype()), IntegerArray)
        assert_type(pd.array([1, 2, np.nan], "UInt16"), IntegerArray)
        # pandas UInt32
        assert_type(pd.array([1, 2, np.nan], pd.UInt32Dtype()), IntegerArray)
        assert_type(pd.array([1, 2, np.nan], "UInt32"), IntegerArray)
        # pandas UInt64
        assert_type(pd.array([1, 2, np.nan], pd.UInt64Dtype()), IntegerArray)
        assert_type(pd.array([1, 2, np.nan], "Int64"), IntegerArray)


def test_constructor() -> None:
    check(
        assert_type(IntegerArray(np.array([1]), np.array([False])), IntegerArray),
        IntegerArray,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        _list_np = IntegerArray([1], np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _np_list = IntegerArray(np.array([1]), [False])  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _pd_arr = IntegerArray(pd.array([1]), np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _i = IntegerArray(pd.Index([1]), np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _s = IntegerArray(pd.Series([1]), np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_dtype_signed() -> None:
    check(assert_type(pd.array([-1]).dtype, IntegerDtype), pd.Int64Dtype)

    check(assert_type(pd.array([-1], "Int8").dtype, IntegerDtype), pd.Int8Dtype)
    check(assert_type(pd.array([-1], "Int16").dtype, IntegerDtype), pd.Int16Dtype)
    check(assert_type(pd.array([-1], "Int32").dtype, IntegerDtype), pd.Int32Dtype)
    check(assert_type(pd.array([-1], "Int64").dtype, IntegerDtype), pd.Int64Dtype)


def test_dtype_unsigned() -> None:
    check(assert_type(pd.array([1], "UInt8").dtype, IntegerDtype), pd.UInt8Dtype)
    check(assert_type(pd.array([1], "UInt16").dtype, IntegerDtype), pd.UInt16Dtype)
    check(assert_type(pd.array([1], "UInt32").dtype, IntegerDtype), pd.UInt32Dtype)
    check(assert_type(pd.array([1], "UInt64").dtype, IntegerDtype), pd.UInt64Dtype)
