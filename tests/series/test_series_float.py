from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import assert_type

from tests import (
    ASTYPE_FLOAT_ARGS,
    TYPE_FLOAT_ARGS,
    FloatDtypeArg,
    PandasAstypeFloatDtypeArg,
    check,
    skip_platform,
)


def test_constructor() -> None:
    check(assert_type(pd.Series([1.0]), "pd.Series[float]"), pd.Series, np.floating)
    check(
        assert_type(pd.Series([1.0, np.float64(1)]), "pd.Series[float]"),
        pd.Series,
        np.floating,
    )
    check(
        assert_type(pd.Series(np.array([1.0], np.float64)), "pd.Series[float]"),
        pd.Series,
        np.floating,
    )
    check(
        assert_type(pd.Series(pd.array([1.0])), "pd.Series[float]"),
        pd.Series,
        np.floating,
    )
    check(
        assert_type(pd.Series(pd.Index([1.0])), "pd.Series[float]"),
        pd.Series,
        np.floating,
    )
    check(
        assert_type(pd.Series(pd.Series([1.0])), "pd.Series[float]"),
        pd.Series,
        np.floating,
    )


@pytest.mark.parametrize(("dtype", "target_dtype"), TYPE_FLOAT_ARGS.items())
def test_constructor_dtype(dtype: FloatDtypeArg, target_dtype: type) -> None:
    def maker() -> "pd.Series[float]":
        return assert_type(pd.Series([1.0], dtype=dtype), "pd.Series[float]")

    skip_platform(maker, dtype)

    check(maker(), pd.Series, target_dtype)
    if TYPE_CHECKING:
        # python float
        assert_type(pd.Series([1.0], dtype=float), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="float"), "pd.Series[float]")
        # pandas Float32
        assert_type(pd.Series([1.0], dtype=pd.Float32Dtype()), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="Float32"), "pd.Series[float]")
        # pandas Float64
        assert_type(pd.Series([1.0], dtype=pd.Float64Dtype()), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="Float64"), "pd.Series[float]")
        # numpy float16
        assert_type(pd.Series([1.0], dtype=np.half), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="half"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="float16"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="e"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="f2"), "pd.Series[float]")
        # numpy float32
        assert_type(pd.Series([1.0], dtype=np.single), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="single"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="float32"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="f"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="f4"), "pd.Series[float]")
        # numpy float64
        assert_type(pd.Series([1.0], dtype=np.double), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="double"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="float64"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="d"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="f8"), "pd.Series[float]")
        # numpy float128
        assert_type(pd.Series([1.0], dtype=np.longdouble), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="longdouble"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="float128"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="g"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="f16"), "pd.Series[float]")
        # pyarrow float32
        assert_type(pd.Series([1.0], dtype="float32[pyarrow]"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="float[pyarrow]"), "pd.Series[float]")
        # pyarrow float64
        assert_type(pd.Series([1.0], dtype="float64[pyarrow]"), "pd.Series[float]")
        assert_type(pd.Series([1.0], dtype="double[pyarrow]"), "pd.Series[float]")


@pytest.mark.parametrize(
    ("cast_arg", "target_type"), ASTYPE_FLOAT_ARGS.items(), ids=repr
)
def test_astype_float(
    cast_arg: FloatDtypeArg | PandasAstypeFloatDtypeArg, target_type: type
) -> None:
    s = pd.Series([1, 2, 3])

    skip_platform(lambda: s.astype(cast_arg), cast_arg)

    check(s.astype(cast_arg), pd.Series, target_type)

    if TYPE_CHECKING:
        # python float
        assert_type(s.astype(float), "pd.Series[float]")
        assert_type(s.astype("float"), "pd.Series[float]")
        # pandas Float32
        assert_type(s.astype(pd.Float32Dtype()), "pd.Series[float]")
        assert_type(s.astype("Float32"), "pd.Series[float]")
        # pandas Float64
        assert_type(s.astype(pd.Float64Dtype()), "pd.Series[float]")
        assert_type(s.astype("Float64"), "pd.Series[float]")
        # numpy float16
        assert_type(s.astype(np.half), "pd.Series[float]")
        assert_type(s.astype("half"), "pd.Series[float]")
        assert_type(s.astype("float16"), "pd.Series[float]")
        assert_type(s.astype("e"), "pd.Series[float]")
        assert_type(s.astype("f2"), "pd.Series[float]")
        # numpy float32
        assert_type(s.astype(np.single), "pd.Series[float]")
        assert_type(s.astype("single"), "pd.Series[float]")
        assert_type(s.astype("float32"), "pd.Series[float]")
        assert_type(s.astype("f"), "pd.Series[float]")
        assert_type(s.astype("f4"), "pd.Series[float]")
        # numpy float64
        assert_type(s.astype(np.double), "pd.Series[float]")
        assert_type(s.astype("double"), "pd.Series[float]")
        assert_type(s.astype("float64"), "pd.Series[float]")
        assert_type(s.astype("d"), "pd.Series[float]")
        assert_type(s.astype("f8"), "pd.Series[float]")
        # numpy float128
        assert_type(s.astype(np.longdouble), "pd.Series[float]")
        assert_type(s.astype("longdouble"), "pd.Series[float]")
        assert_type(s.astype("float128"), "pd.Series[float]")
        assert_type(s.astype("g"), "pd.Series[float]")
        assert_type(s.astype("f16"), "pd.Series[float]")
        # pyarrow float32
        assert_type(s.astype("float32[pyarrow]"), "pd.Series[float]")
        assert_type(s.astype("float[pyarrow]"), "pd.Series[float]")
        # pyarrow float64
        assert_type(s.astype("float64[pyarrow]"), "pd.Series[float]")
        assert_type(s.astype("double[pyarrow]"), "pd.Series[float]")
