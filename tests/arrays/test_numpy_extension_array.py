import numpy as np
import pandas as pd
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    BuiltinDtypeArg,
    NumpyNotTimeDtypeArg,
    check,
    exception_on_platform,
    get_dtype,
)


def test_constructor() -> None:
    # check(
    #     assert_type(pd.array([pd.NA, None]), NumpyExtensionArray), NumpyExtensionArray
    # )

    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd.array([1, "🐼"]), NumpyExtensionArray
        ),
        NumpyExtensionArray,
    )
    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd.array(np.array([1, "🐼"], np.object_)), NumpyExtensionArray
        ),
        NumpyExtensionArray,
    )
    # check(
    #     assert_type(pd.array(pd.array([pd.NA, None])), NumpyExtensionArray),
    #     NumpyExtensionArray,
    # )
    check(
        assert_type(pd.array(pd.RangeIndex(0, 1)), NumpyExtensionArray),
        NumpyExtensionArray,
    )


@pytest.mark.parametrize("dtype", get_dtype(BuiltinDtypeArg | NumpyNotTimeDtypeArg))
def test_constructor_dtype(dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg):
    if dtype == "V" or "void" in str(dtype):
        check(
            assert_type(pd.array([b"1"], dtype=dtype), NumpyExtensionArray),
            NumpyExtensionArray,
        )
    else:
        exc = exception_on_platform(dtype)
        if exc:
            with pytest.raises(exc):
                pd.array([1], dtype=dtype)
        else:
            check(
                assert_type(pd.array([1], dtype=dtype), NumpyExtensionArray),
                NumpyExtensionArray,
            )
