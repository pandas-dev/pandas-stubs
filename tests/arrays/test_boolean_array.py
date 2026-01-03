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
from pandas.core.arrays.boolean import BooleanArray
import pytest
from typing_extensions import assert_type

from pandas._libs.missing import NAType

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
)
from tests._typing import PandasBooleanDtypeArg
from tests.dtypes import PANDAS_BOOL_ARGS
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("data", powerset([True, np.bool_(True)], 1))
@pytest.mark.parametrize("missing_values", powerset([np.nan, None, pd.NA]))
def test_construction_sequence(
    data: tuple[bool | np.bool_, ...],
    missing_values: tuple[Any, ...],
    typ: Callable[[Sequence[Any]], Sequence[Any]],
) -> None:
    check(pd.array(typ([*data, *missing_values])), BooleanArray)
    check(pd.array(typ([False, *data, *missing_values])), BooleanArray)
    check(pd.array(typ([np.bool_(False), *data, *missing_values])), BooleanArray)

    if TYPE_CHECKING:
        assert_type(pd.array([False, True]), BooleanArray)
        assert_type(pd.array([np.bool_(True), np.bool_(False)]), BooleanArray)

        assert_type(pd.array([True, np.bool_(True)]), BooleanArray)

        assert_type(pd.array([True, np.bool_(True), None]), BooleanArray)
        assert_type(pd.array([True, np.bool_(True), pd.NA]), BooleanArray)

        assert_type(pd.array([True, np.bool_(True), None, pd.NA]), BooleanArray)

        assert_type(pd.array((True, np.bool_(True))), BooleanArray)
        assert_type(pd.array((True, np.bool_(True), pd.NA)), BooleanArray)

        assert_type(pd.array(UserList([True, np.bool_(True)])), BooleanArray)


def test_construction_array_like() -> None:
    np_arr = np.array([True, np.bool_(True)], np.bool_)
    check(assert_type(pd.array(np_arr), BooleanArray), BooleanArray)

    check(
        assert_type(pd.array(pd.array([True, np.bool_(True)])), BooleanArray),
        BooleanArray,
    )


@pytest.mark.parametrize("data", powerset([False, np.bool(False)]))
@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_BOOL_ARGS.items())
def test_construction_dtype(
    data: tuple[bool | np.bool, ...], dtype: PandasBooleanDtypeArg, target_dtype: type
) -> None:
    dtype_notna = target_dtype if data else None
    check(pd.array([*data], dtype), BooleanArray, dtype_notna)
    check(pd.array([True, *data], dtype), BooleanArray, dtype_notna)
    check(pd.array([np.bool(True), *data], dtype), BooleanArray, dtype_notna)

    dtype_na = target_dtype if data else NAType
    check(pd.array([*data, np.nan], dtype), BooleanArray, dtype_na)
    check(pd.array([True, *data, np.nan], dtype), BooleanArray, target_dtype)
    check(pd.array([np.bool(True), *data, np.nan], dtype), BooleanArray, target_dtype)

    if TYPE_CHECKING:
        assert_type(pd.array([], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([], "boolean"), BooleanArray)

        assert_type(pd.array([np.nan], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([np.nan], "boolean"), BooleanArray)

        assert_type(pd.array([False], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([False], "boolean"), BooleanArray)

        assert_type(pd.array([False, True], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([False, True], "boolean"), BooleanArray)

        assert_type(pd.array([False, np.nan], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([False, np.nan], "boolean"), BooleanArray)

        assert_type(pd.array([np.bool(False)], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([np.bool(False)], "boolean"), BooleanArray)

        assert_type(
            pd.array([np.bool(False), np.bool(True)], pd.BooleanDtype()), BooleanArray
        )
        assert_type(pd.array([np.bool(False), np.bool(True)], "boolean"), BooleanArray)

        assert_type(pd.array([np.bool(False), np.nan], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([np.bool(False), np.nan], "boolean"), BooleanArray)

        assert_type(pd.array([False, np.bool(True)], pd.BooleanDtype()), BooleanArray)
        assert_type(pd.array([np.bool(False), True], "boolean"), BooleanArray)


def test_constructor() -> None:
    check(
        assert_type(BooleanArray(np.array([True]), np.array([False])), BooleanArray),
        BooleanArray,
    )

    if TYPE_CHECKING_INVALID_USAGE:
        _list_np = BooleanArray([True], np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _np_list = BooleanArray(np.array([True]), [False])  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _pd_arr = BooleanArray(pd.array([True]), np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _i = BooleanArray(pd.Index([False]), np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _s = BooleanArray(pd.Series([True]), np.array([False]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]


def test_dtype() -> None:
    check(assert_type(pd.array([True]).dtype, pd.BooleanDtype), pd.BooleanDtype)
