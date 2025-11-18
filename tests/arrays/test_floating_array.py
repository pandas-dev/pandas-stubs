from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.arrays.floating import FloatingArray
import pytest
from typing_extensions import assert_type

from tests import (
    PANDAS_FLOAT_ARGS,
    PandasFloatDtypeArg,
    check,
    skip_platform,
)


def test_constructor() -> None:
    check(assert_type(pd.array([1.0]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([1.0, np.float64(1)]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([1.0, None]), FloatingArray), FloatingArray)
    check(assert_type(pd.array([1.0, pd.NA, None]), FloatingArray), FloatingArray)

    check(
        assert_type(  # type: ignore[assert-type] # I do not understand
            pd.array(np.array([1.0], np.float64)), FloatingArray
        ),
        FloatingArray,
    )

    check(assert_type(pd.array(pd.array([1.0])), FloatingArray), FloatingArray)


@pytest.mark.parametrize(("dtype", "target_dtype"), PANDAS_FLOAT_ARGS.items(), ids=repr)
def test_constructor_dtype(dtype: PandasFloatDtypeArg, target_dtype: type) -> None:
    def maker() -> FloatingArray:
        return assert_type(pd.array([1.0], dtype=dtype), FloatingArray)

    skip_platform(maker, dtype)

    check(maker(), FloatingArray, target_dtype)

    if TYPE_CHECKING:
        # pandas Float32
        assert_type(pd.array([1.0], dtype=pd.Float32Dtype()), FloatingArray)
        assert_type(pd.array([1.0], dtype="Float32"), FloatingArray)
        # pandas Float64
        assert_type(pd.array([1.0], dtype=pd.Float64Dtype()), FloatingArray)
        assert_type(pd.array([1.0], dtype="Float64"), FloatingArray)
