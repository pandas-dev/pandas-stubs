from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.arrays.boolean import BooleanArray
import pytest
from typing_extensions import assert_type

from tests import (
    check,
    get_dtype,
)

if TYPE_CHECKING:
    from pandas._typing import PandasBooleanDtypeArg
else:
    from tests._typing import PandasBooleanDtypeArg


def test_constructor() -> None:
    check(assert_type(pd.array([True]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([True, np.bool(True)]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([True, None]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([True, pd.NA]), BooleanArray), BooleanArray)

    check(assert_type(pd.array(np.array([1], np.bool_)), BooleanArray), BooleanArray)

    check(assert_type(pd.array(pd.array([True])), BooleanArray), BooleanArray)

    pd.array([True], dtype=pd.BooleanDtype())


@pytest.mark.parametrize("dtype", get_dtype(PandasBooleanDtypeArg))
def test_constructor_dtype(dtype: PandasBooleanDtypeArg) -> None:
    check(assert_type(pd.array([True], dtype=dtype), BooleanArray), BooleanArray)
