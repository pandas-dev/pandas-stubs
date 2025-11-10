import numpy as np
import pandas as pd
from pandas.core.arrays.boolean import BooleanArray
from typing_extensions import assert_type

from tests import check


def test_constructor() -> None:
    check(assert_type(pd.array([True]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([True, np.bool(True)]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([True, None]), BooleanArray), BooleanArray)
    check(assert_type(pd.array([True, pd.NA]), BooleanArray), BooleanArray)

    check(assert_type(pd.array(np.array([1], np.bool_)), BooleanArray), BooleanArray)

    check(assert_type(pd.array(pd.array([True])), BooleanArray), BooleanArray)
