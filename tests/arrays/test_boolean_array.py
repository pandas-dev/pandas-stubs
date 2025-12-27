from collections import UserList
from typing import (
    Literal,
    cast,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.boolean import BooleanArray
from typing_extensions import assert_type

from tests import check


def test_construction_sequence() -> None:
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[bool | np.bool[Literal[True]]]", [True, np.bool_(True)]
    )

    check(assert_type(pd.array(data), BooleanArray), BooleanArray)
    check(assert_type(pd.array([*data, None]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([*data, pd.NA]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([*data, None, pd.NA]), BooleanArray), BooleanArray)

    check(assert_type(pd.array(tuple(data)), BooleanArray), BooleanArray)
    check(assert_type(pd.array(UserList(data)), BooleanArray), BooleanArray)


def test_construction_array_like() -> None:
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[bool | np.bool[Literal[True]]]", [True, np.bool_(True)]
    )
    np_arr = np.array(data, np.bool_)

    check(assert_type(pd.array(np_arr), BooleanArray), BooleanArray)

    check(assert_type(pd.array(pd.array(data)), BooleanArray), BooleanArray)


def test_construction_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "boolean"), BooleanArray), BooleanArray)
