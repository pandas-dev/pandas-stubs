from collections import UserList
from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import pandas as pd
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.numpy_ import NumpyExtensionArray
import pytest
from typing_extensions import assert_type

from tests import (
    PD_LTE_23,
    TYPE_CHECKING_INVALID_USAGE,
    check,
    exception_on_platform,
)
from tests._typing import (
    BuiltinNotStrDtypeArg,
    NumpyNotTimeDtypeArg,
    np_ndarray,
)
from tests.dtypes import (
    NUMPY_NOT_DATETIMELIKE_DTYPE_ARGS,
    PYTHON_NOT_STR_DTYPE_ARGS,
)
from tests.utils import powerset


@pytest.mark.parametrize("typ", [list, tuple, UserList])
@pytest.mark.parametrize("missing_values", powerset([None, pd.NA, pd.NaT], 1))
def test_construction_sequence(
    missing_values: tuple[Any, ...], typ: Callable[[Sequence[Any]], Sequence[Any]]
) -> None:
    # `pd.NaT in [pd.NA, pd.NaT]` leads to an exception
    if missing_values[-1] is pd.NaT:
        expected_type = DatetimeArray if PD_LTE_23 else NumpyExtensionArray
        check(pd.array(typ(missing_values)), expected_type)
        check(pd.array(typ((np.nan, *missing_values))), expected_type)
    else:
        check(pd.array(typ(missing_values)), NumpyExtensionArray)

    if TYPE_CHECKING:
        assert_type(pd.array([None]), NumpyExtensionArray)
        assert_type(pd.array([pd.NA]), NumpyExtensionArray)
        assert_type(pd.array([pd.NaT]), NumpyExtensionArray)

        # mypy infers any list with np.nan, which is a float, and types like
        # pd.NA and None to be list[object]
        # It would be quite unusual for user code to be mixing np.nan with the
        # other "NA"-like types.
        assert_type(pd.array([np.nan, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]
        assert_type(pd.array([None, pd.NA]), NumpyExtensionArray)
        assert_type(pd.array([None, pd.NaT]), NumpyExtensionArray)
        assert_type(pd.array([pd.NA, pd.NaT]), NumpyExtensionArray)

        assert_type(pd.array([np.nan, None, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]
        assert_type(pd.array([np.nan, pd.NA, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]
        assert_type(pd.array([None, pd.NA, pd.NaT]), NumpyExtensionArray)

        assert_type(pd.array([np.nan, None, pd.NA, pd.NaT]), NumpyExtensionArray)  # type: ignore[assert-type]

        assert_type(pd.array((np.nan, None, pd.NA, pd.NaT)), NumpyExtensionArray)

        assert_type(pd.array(UserList([np.nan, pd.NA, pd.NaT])), NumpyExtensionArray)  # type: ignore[assert-type]


def test_construction_array_like() -> None:
    data = [1, b"a"]
    np_arr = np.array(data, np.object_)

    check(assert_type(pd.array(data), NumpyExtensionArray), NumpyExtensionArray)

    check(assert_type(pd.array(np_arr), NumpyExtensionArray), NumpyExtensionArray)

    check(
        assert_type(pd.array(pd.array(data)), NumpyExtensionArray), NumpyExtensionArray
    )

    check(
        assert_type(pd.array(pd.RangeIndex(0, 1)), NumpyExtensionArray),
        NumpyExtensionArray,
    )


def test_construction_dtype_nan() -> None:
    check(
        assert_type(pd.array([np.nan], float), NumpyExtensionArray),
        NumpyExtensionArray,
    )


@pytest.mark.parametrize(
    ("dtype", "target_dtype"),
    (PYTHON_NOT_STR_DTYPE_ARGS | NUMPY_NOT_DATETIMELIKE_DTYPE_ARGS).items(),
)
def test_construction_dtype(
    dtype: BuiltinNotStrDtypeArg | NumpyNotTimeDtypeArg, target_dtype: type
) -> None:
    exc = exception_on_platform(dtype)
    if exc:
        with pytest.raises(exc, match=rf"data type {dtype!r} not understood"):
            assert_type(pd.array([1], dtype=dtype), NumpyExtensionArray)
    elif dtype == "V" or "void" in str(dtype):
        check(pd.array([b"1"], dtype=dtype), NumpyExtensionArray, target_dtype)
    else:
        check(pd.array([1], dtype=dtype), NumpyExtensionArray, target_dtype)

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
        # python bytes
        assert_type(pd.array([1], dtype=bytes), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="bytes"), NumpyExtensionArray)
        # python object
        assert_type(pd.array([1], dtype=object), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="object"), NumpyExtensionArray)

        # numpy boolean
        assert_type(pd.array([1], dtype=np.bool_), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="?"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="b1"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="bool_"), NumpyExtensionArray)
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
        # numpy long
        assert_type(pd.array([1], dtype="long"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="l"), NumpyExtensionArray)
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
        # numpy ulong
        assert_type(pd.array([1], dtype="ulong"), NumpyExtensionArray)
        assert_type(pd.array([1], dtype="L"), NumpyExtensionArray)
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
        assert_type(pd.array([1], dtype="<f2"), NumpyExtensionArray)
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
        assert_type(pd.array([1], dtype="O"), NumpyExtensionArray)
        # numpy void
        assert_type(pd.array([b"1"], dtype=np.void), NumpyExtensionArray)
        assert_type(pd.array([b"1"], dtype="void"), NumpyExtensionArray)


@pytest.mark.parametrize("creator", [np.array, pd.array])
def test_constructor(creator: Callable[..., np_ndarray | NumpyExtensionArray]) -> None:
    check(NumpyExtensionArray(creator([None])), NumpyExtensionArray)

    if TYPE_CHECKING:
        assert_type(NumpyExtensionArray(np.array([1])), NumpyExtensionArray)
        assert_type(NumpyExtensionArray(pd.array([None])), NumpyExtensionArray)

    if TYPE_CHECKING_INVALID_USAGE:
        _list = NumpyExtensionArray([1])  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _tuple = NumpyExtensionArray((1,))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _i = NumpyExtensionArray(pd.Index([1]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
        _s = NumpyExtensionArray(pd.Series([1]))  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
