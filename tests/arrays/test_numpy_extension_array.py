import numpy as np
import pandas as pd
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    check,
    get_dtype,
)
from tests._typing import (
    BuiltinDtypeArg,
    NumpyNotTimeDtypeArg,
)


def test_constructor() -> None:
    # check(
    #     assert_type(pd.array([pd.NA, None]), NumpyExtensionArray), NumpyExtensionArray
    # )

    pd_arr = pd.array([1, "🐼"])
    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd_arr, NumpyExtensionArray
        ),
        NumpyExtensionArray,
    )

    np_arr = np.array([1, "🐼"], np.object_)
    check(assert_type(pd.array(np_arr), NumpyExtensionArray), NumpyExtensionArray)
    # check(
    #     assert_type(pd.array(pd.array([pd.NA, None])), NumpyExtensionArray),
    #     NumpyExtensionArray,
    # )
    check(
        assert_type(pd.array(pd.RangeIndex(0, 1)), NumpyExtensionArray),
        NumpyExtensionArray,
    )


@pytest.mark.parametrize("dtype", get_dtype(BuiltinDtypeArg | NumpyNotTimeDtypeArg))
def test_constructors_dtype(dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg):
    if dtype == "V" or "void" in str(dtype):
        check(
            assert_type(pd.array([b"1"], dtype=dtype), NumpyExtensionArray),
            NumpyExtensionArray,
        )
    else:
        check(
            assert_type(pd.array([1], dtype=dtype), NumpyExtensionArray),
            NumpyExtensionArray,
        )
