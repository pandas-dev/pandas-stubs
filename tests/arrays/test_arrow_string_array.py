from typing import Literal

import numpy as np
import pandas as pd
from pandas.core.arrays.string_arrow import ArrowStringArray
from typing_extensions import assert_type

from tests import check


def test_construction_dtype_na() -> None:
    check(
        assert_type(pd.array([np.nan], "string[pyarrow]"), ArrowStringArray),
        ArrowStringArray,
    )


def test_dtype() -> None:
    arr = pd.array(["a"], "string[pyarrow]")
    check(assert_type(arr.dtype, 'pd.StringDtype[Literal["pyarrow"]]'), pd.StringDtype)
    assert assert_type(arr.dtype.storage, Literal["pyarrow"]) == "pyarrow"
