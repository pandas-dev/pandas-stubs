import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringArray
from typing_extensions import assert_type

from tests import check


def test_construction_dtype_na() -> None:
    check(assert_type(pd.array([np.nan], "string[python]"), StringArray), StringArray)
