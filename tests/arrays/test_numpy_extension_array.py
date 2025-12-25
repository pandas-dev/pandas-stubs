from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    check,
    exception_on_platform,
)
from tests._typing import (
    BuiltinDtypeArg,
    NumpyNotTimeDtypeArg,
)
from tests.dtypes import (
    NUMPY_NOT_DATETIMELIKE_DTYPE_ARGS,
    PYTHON_DTYPE_ARGS,
)


def test_constructor() -> None:
    none_data = [pd.NA, None]
    check(assert_type(pd.array(none_data), NumpyExtensionArray), NumpyExtensionArray)

    mixed_data = [1, "🐼"]
    check(assert_type(pd.array(mixed_data), NumpyExtensionArray), NumpyExtensionArray)

    np_arr = np.array(mixed_data, np.object_)
    check(assert_type(pd.array(np_arr), NumpyExtensionArray), NumpyExtensionArray)
    check(
        assert_type(pd.array(pd.array(none_data)), NumpyExtensionArray),
        NumpyExtensionArray,
    )
    check(
        assert_type(pd.array(pd.RangeIndex(0, 1)), NumpyExtensionArray),
        NumpyExtensionArray,
    )


@pytest.mark.parametrize(
    ("dtype", "target_dtype"),
    (PYTHON_DTYPE_ARGS | NUMPY_NOT_DATETIMELIKE_DTYPE_ARGS).items(),
)
def test_constructor_dtype(
    dtype: BuiltinDtypeArg | NumpyNotTimeDtypeArg, target_dtype: type
) -> None:
    data = [b"1" if dtype == "V" or "void" in str(dtype) else 1]
    exc = exception_on_platform(dtype)
    if exc:
        with pytest.raises(exc, match=rf"data type {dtype!r} not understood"):
            assert_type(pd.array(data, dtype=dtype), NumpyExtensionArray)
    else:
        check(pd.array(data, dtype=dtype), NumpyExtensionArray, target_dtype)

    if TYPE_CHECKING:
        # python boolean
        assert_type(pd.array([1], dtype=bool), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="bool"), NumpyExtensionArray)
        # python int
        assert_type(pd.array([1], dtype=int), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="int"), NumpyExtensionArray)
        # python float
        assert_type(pd.array([1], dtype=float), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="float"), NumpyExtensionArray)
        # python complex
        assert_type(pd.array([1], dtype=complex), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="complex"), NumpyExtensionArray)
        # python string
        assert_type(pd.array([1], dtype=str), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="str"), NumpyExtensionArray)
        # python bytes
        assert_type(pd.array([1], dtype=bytes), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="bytes"), NumpyExtensionArray)

        # numpy boolean
        assert_type(pd.array([1], dtype=np.bool_), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="bool_"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="?"), NumpyExtensionArray)
        # numpy int8
        assert_type(pd.array([1], dtype=np.byte), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="byte"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="b"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="int8"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="i1"), NumpyExtensionArray)
        # numpy int16
        assert_type(pd.array([1], dtype=np.short), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="short"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="h"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="int16"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="i2"), NumpyExtensionArray)
        # numpy int32
        assert_type(pd.array([1], dtype=np.intc), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="intc"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="i"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="int32"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="i4"), NumpyExtensionArray)
        # numpy int64
        assert_type(pd.array([1], dtype=np.int_), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="int_"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="int64"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="i8"), NumpyExtensionArray)
        # numpy extended int
        assert_type(pd.array([1], dtype=np.longlong), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="longlong"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="q"), NumpyExtensionArray)
        # numpy signed pointer
        assert_type(pd.array([1], dtype=np.intp), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="intp"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="p"), NumpyExtensionArray)
        # numpy uint8
        assert_type(pd.array([1], dtype=np.ubyte), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="ubyte"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="B"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uint8"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="u1"), NumpyExtensionArray)
        # numpy uint16
        assert_type(pd.array([1], dtype=np.ushort), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="ushort"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="H"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uint16"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="u2"), NumpyExtensionArray)
        # numpy uint32
        assert_type(pd.array([1], dtype=np.uintc), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uintc"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="I"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uint32"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="u4"), NumpyExtensionArray)
        # numpy uint64
        assert_type(pd.array([1], dtype=np.uint), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uint"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uint64"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="u8"), NumpyExtensionArray)
        # numpy extended uint
        assert_type(pd.array([1], dtype=np.ulonglong), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="ulonglong"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="Q"), NumpyExtensionArray)
        # numpy unsigned pointer
        assert_type(pd.array([1], dtype=np.uintp), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="uintp"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="P"), NumpyExtensionArray)
        # numpy float16
        assert_type(pd.array([1], dtype=np.half), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="half"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="e"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="float16"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="f2"), NumpyExtensionArray)
        # numpy float32
        assert_type(pd.array([1], dtype=np.single), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="single"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="f"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="float32"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="f4"), NumpyExtensionArray)
        # numpy float64
        assert_type(pd.array([1], dtype=np.double), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="double"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="d"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="float64"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="f8"), NumpyExtensionArray)
        # numpy float128
        assert_type(pd.array([1], dtype=np.longdouble), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="g"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="longdouble"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="f16"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="float128"), NumpyExtensionArray)
        # numpy complex64
        assert_type(pd.array([1], dtype=np.csingle), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="csingle"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="F"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="complex64"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="c8"), NumpyExtensionArray)
        # numpy complex128
        assert_type(pd.array([1], dtype=np.cdouble), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="cdouble"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="D"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="complex128"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="c16"), NumpyExtensionArray)
        # numpy complex256
        assert_type(pd.array([1], dtype=np.clongdouble), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="clongdouble"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="G"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="c32"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="complex256"), NumpyExtensionArray)
        # numpy string
        assert_type(pd.array([1], dtype=np.str_), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="str_"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="unicode"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="U"), NumpyExtensionArray)
        # numpy bytes
        assert_type(pd.array([1], dtype=np.bytes_), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="bytes_"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="S"), NumpyExtensionArray)
        # numpy object
        assert_type(pd.array([1], dtype=np.object_), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="object_"), NumpyExtensionArray)
        # numpy void
        assert_type(pd.array([b"1"], dtype=np.void), NumpyExtensionArray)
        assert_type(pd.array([b"1"], dtype="void"), NumpyExtensionArray)
