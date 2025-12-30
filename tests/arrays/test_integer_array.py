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
from pandas.core.arrays.integer import IntegerArray
import pytest
from typing_extensions import assert_type

from tests import check
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


def test_construction_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "Int8"), IntegerArray), IntegerArray)
