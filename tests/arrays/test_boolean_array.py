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

from tests import check
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


def test_construction_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "boolean"), BooleanArray), BooleanArray)
