from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typing_extensions import (
    Never,
    assert_type,
)

from tests import (
    TYPE_CHECKING_INVALID_USAGE,
    check,
    exception_on_platform,
)
from tests.dtypes import ASTYPE_FLOAT_NOT_NUMPY16_ARGS

if TYPE_CHECKING:
    from pandas.core.indexes.base import FloatNotNumpy16DtypeArg


def test_constructor() -> None:
    check(assert_type(pd.Index([1.0]), "pd.Index[float]"), pd.Index, np.floating)
    check(
        assert_type(pd.Index([1.0, np.float64(1)]), "pd.Index[float]"),
        pd.Index,
        np.floating,
    )
    nparr = np.array([1.0], np.float64)
    check(assert_type(pd.Index(nparr), "pd.Index[float]"), pd.Index, np.floating)
    check(
        assert_type(pd.Index(pd.array([1.0])), "pd.Index[float]"),
        pd.Index,
        np.floating,
    )
    check(
        assert_type(pd.Index(pd.Index([1.0])), "pd.Index[float]"),
        pd.Index,
        np.floating,
    )
    check(
        assert_type(pd.Index(pd.Series([1.0])), "pd.Index[float]"),
        pd.Index,
        np.floating,
    )


@pytest.mark.parametrize(
    ("dtype", "target_dtype"), ASTYPE_FLOAT_NOT_NUMPY16_ARGS.items()
)
def test_constructor_dtype(
    dtype: "FloatNotNumpy16DtypeArg", target_dtype: type
) -> None:
    exc = exception_on_platform(dtype)
    if exc:
        with pytest.raises(exc, match=rf"data type {dtype!r} not understood"):
            assert_type(pd.Index([1.0], dtype=dtype), "pd.Index[float]")
    else:
        check(pd.Index([1.0], dtype=dtype), pd.Index, target_dtype)

    if TYPE_CHECKING:
        # python float
        assert_type(pd.Index([1.0], dtype=float), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="float"), "pd.Index[float]")
        # pandas Float32
        assert_type(pd.Index([1.0], dtype=pd.Float32Dtype()), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="Float32"), "pd.Index[float]")
        # pandas Float64
        assert_type(pd.Index([1.0], dtype=pd.Float64Dtype()), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="Float64"), "pd.Index[float]")
        # numpy float32
        assert_type(pd.Index([1.0], dtype=np.single), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="single"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="float32"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="f"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="f4"), "pd.Index[float]")
        # numpy float64
        assert_type(pd.Index([1.0], dtype=np.double), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="double"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="float64"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="d"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="f8"), "pd.Index[float]")
        # numpy float128
        assert_type(pd.Index([1.0], dtype=np.longdouble), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="longdouble"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="float128"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="g"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="f16"), "pd.Index[float]")
        # pyarrow float16
        assert_type(pd.Index([1.0], dtype="float16[pyarrow]"), "pd.Index[float]")
        # pyarrow float32
        assert_type(pd.Index([1.0], dtype="float32[pyarrow]"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="float[pyarrow]"), "pd.Index[float]")
        # pyarrow float64
        assert_type(pd.Index([1.0], dtype="float64[pyarrow]"), "pd.Index[float]")
        assert_type(pd.Index([1.0], dtype="double[pyarrow]"), "pd.Index[float]")

    if TYPE_CHECKING_INVALID_USAGE:
        # numpy float16
        def _0() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(pd.Index([1.0], dtype=np.half), Never)

        def _1() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(pd.Index([1.0], dtype="half"), Never)

        def _2() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(pd.Index([1.0], dtype="float16"), Never)

        def _3() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(pd.Index([1.0], dtype="e"), Never)

        def _4() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(pd.Index([1.0], dtype="f2"), Never)


@pytest.mark.parametrize(
    ("cast_arg", "target_type"), ASTYPE_FLOAT_NOT_NUMPY16_ARGS.items(), ids=repr
)
def test_astype_float(cast_arg: "FloatNotNumpy16DtypeArg", target_type: type) -> None:
    s = pd.Index([1, 2, 3])

    exc = exception_on_platform(cast_arg)
    if exc:
        with pytest.raises(exc, match=rf"data type {cast_arg!r} not understood"):
            assert_type(s.astype(cast_arg), "pd.Index[float]")
    else:
        check(s.astype(cast_arg), pd.Index, target_type)

    if TYPE_CHECKING:
        # python float
        assert_type(s.astype(float), "pd.Index[float]")
        assert_type(s.astype("float"), "pd.Index[float]")
        # pandas Float32
        assert_type(s.astype(pd.Float32Dtype()), "pd.Index[float]")
        assert_type(s.astype("Float32"), "pd.Index[float]")
        # pandas Float64
        assert_type(s.astype(pd.Float64Dtype()), "pd.Index[float]")
        assert_type(s.astype("Float64"), "pd.Index[float]")
        # numpy float32
        assert_type(s.astype(np.single), "pd.Index[float]")
        assert_type(s.astype("single"), "pd.Index[float]")
        assert_type(s.astype("float32"), "pd.Index[float]")
        assert_type(s.astype("f"), "pd.Index[float]")
        assert_type(s.astype("f4"), "pd.Index[float]")
        # numpy float64
        assert_type(s.astype(np.double), "pd.Index[float]")
        assert_type(s.astype("double"), "pd.Index[float]")
        assert_type(s.astype("float64"), "pd.Index[float]")
        assert_type(s.astype("d"), "pd.Index[float]")
        assert_type(s.astype("f8"), "pd.Index[float]")
        # numpy float128
        assert_type(s.astype(np.longdouble), "pd.Index[float]")
        assert_type(s.astype("longdouble"), "pd.Index[float]")
        assert_type(s.astype("float128"), "pd.Index[float]")
        assert_type(s.astype("g"), "pd.Index[float]")
        assert_type(s.astype("f16"), "pd.Index[float]")
        # pyarrow float16
        assert_type(s.astype("float16[pyarrow]"), "pd.Index[float]")
        # pyarrow float32
        assert_type(s.astype("float32[pyarrow]"), "pd.Index[float]")
        assert_type(s.astype("float[pyarrow]"), "pd.Index[float]")
        # pyarrow float64
        assert_type(s.astype("float64[pyarrow]"), "pd.Index[float]")
        assert_type(s.astype("double[pyarrow]"), "pd.Index[float]")

    if TYPE_CHECKING_INVALID_USAGE:

        def _0() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(s.astype(np.half), Never)

        def _1() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(s.astype("half"), Never)

        def _2() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(s.astype("float16"), Never)

        def _3() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(s.astype("e"), Never)

        def _4() -> None:  # pyright: ignore[reportUnusedFunction]
            assert_type(s.astype("f2"), Never)
