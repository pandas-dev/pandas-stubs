from collections import UserList
from typing import cast

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
from typing_extensions import assert_type

from tests import check


def test_constructor_sequence() -> None:
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[str | np.str_]", ["1", np.str_("1")]
    )

    check(assert_type(pd.array(data), StringArray), StringArray)
    check(assert_type(pd.array([*data, None]), StringArray), StringArray)
    check(assert_type(pd.array([*data, pd.NA]), StringArray), StringArray)
    check(assert_type(pd.array([*data, None, pd.NA]), StringArray), StringArray)

    check(assert_type(pd.array(tuple(data)), StringArray), StringArray)
    check(assert_type(pd.array(UserList(data)), StringArray), StringArray)


def test_constructor_array_like() -> None:
    data = cast(  # pyright: ignore[reportUnnecessaryCast]
        "list[str | np.str_]", ["1", np.str_("1")]
    )
    np_arr = np.array(data, np.str_)

    check(assert_type(pd.array(np_arr), StringArray), StringArray)

    check(assert_type(pd.array(pd.array(data)), StringArray), StringArray)


def test_constructor_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "string"), StringArray), StringArray)
