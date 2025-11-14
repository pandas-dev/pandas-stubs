import numpy as np
import pandas as pd
from pandas.core.arrays.integer import IntegerArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    check(assert_type(pd.array([1]), IntegerArray), IntegerArray)
    check(assert_type(pd.array([1, np.int64(1)]), IntegerArray), IntegerArray)
    check(assert_type(pd.array([1, None]), IntegerArray), IntegerArray)
    check(assert_type(pd.array([1, pd.NA, None]), IntegerArray), IntegerArray)

    np_arr = np.array([1], np.int64)
    check(assert_type(pd.array(np_arr), IntegerArray), IntegerArray)

    check(assert_type(pd.array(pd.array([1])), IntegerArray), IntegerArray)
