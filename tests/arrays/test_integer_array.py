from collections import UserList
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.integer import IntegerArray
from typing_extensions import assert_type

from tests import check

if TYPE_CHECKING:
    from numpy._typing import _8Bit


def test_constructor_sequence() -> None:
    data = cast(
        "list[int | np.signedinteger[_8Bit]]", [1, np.int8(1)]
    )  # pyright: ignore[reportUnnecessaryCast]

    check(assert_type(pd.array(data), IntegerArray), IntegerArray)
    check(assert_type(pd.array([*data, None]), IntegerArray), IntegerArray)
    check(assert_type(pd.array([*data, pd.NA]), IntegerArray), IntegerArray)
    check(assert_type(pd.array([*data, None, pd.NA]), IntegerArray), IntegerArray)

    check(assert_type(pd.array(tuple(data)), IntegerArray), IntegerArray)
    check(assert_type(pd.array(UserList(data)), IntegerArray), IntegerArray)


def test_constructor_array_like() -> None:
    data = cast(
        "list[int | np.signedinteger[_8Bit]]", [1, np.int8(1)]
    )  # pyright: ignore[reportUnnecessaryCast]
    np_arr = np.array(data, np.int8)

    check(assert_type(pd.array(np_arr), IntegerArray), IntegerArray)

    check(assert_type(pd.array(pd.array(data)), IntegerArray), IntegerArray)


def test_constructor_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "Int8"), IntegerArray), IntegerArray)
