from __future__ import annotations

from contextlib import (
    AbstractContextManager,
    nullcontext,
    suppress,
)
import sys
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

# Next set of imports is to keep the private imports needed for testing
# in one place
from pandas._testing import ensure_clean as ensure_clean
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.util.version import Version
import pytest

from pandas.core.dtypes.base import ExtensionDtype

if TYPE_CHECKING:
    from pandas._typing import (
        BooleanDtypeArg as BooleanDtypeArg,
        BuiltinDtypeArg as BuiltinDtypeArg,
        BytesDtypeArg as BytesDtypeArg,
        CategoryDtypeArg as CategoryDtypeArg,
        ComplexDtypeArg as ComplexDtypeArg,
        Dtype as Dtype,
        FloatDtypeArg as FloatDtypeArg,
        IntDtypeArg as IntDtypeArg,
        NumpyFloat16DtypeArg as NumpyFloat16DtypeArg,
        NumpyNotTimeDtypeArg as NumpyNotTimeDtypeArg,
        ObjectDtypeArg as ObjectDtypeArg,
        PandasAstypeComplexDtypeArg as PandasAstypeComplexDtypeArg,
        PandasAstypeFloatDtypeArg as PandasAstypeFloatDtypeArg,
        PandasAstypeTimedeltaDtypeArg as PandasAstypeTimedeltaDtypeArg,
        PandasAstypeTimestampDtypeArg as PandasAstypeTimestampDtypeArg,
        PandasBooleanDtypeArg as PandasBooleanDtypeArg,
        PandasFloatDtypeArg as PandasFloatDtypeArg,
        StrDtypeArg as StrDtypeArg,
        T as T,
        TimedeltaDtypeArg as TimedeltaDtypeArg,
        TimestampDtypeArg as TimestampDtypeArg,
        UIntDtypeArg as UIntDtypeArg,
        VoidDtypeArg as VoidDtypeArg,
        np_1darray as np_1darray,
        np_1darray_anyint as np_1darray_anyint,
        np_1darray_bool as np_1darray_bool,
        np_1darray_bytes as np_1darray_bytes,
        np_1darray_complex as np_1darray_complex,
        np_1darray_dt as np_1darray_dt,
        np_1darray_float as np_1darray_float,
        np_1darray_int64 as np_1darray_int64,
        np_1darray_intp as np_1darray_intp,
        np_1darray_object as np_1darray_object,
        np_1darray_str as np_1darray_str,
        np_1darray_td as np_1darray_td,
        np_2darray as np_2darray,
        np_ndarray as np_ndarray,
        np_ndarray_bool as np_ndarray_bool,
        np_ndarray_dt as np_ndarray_dt,
        np_ndarray_int as np_ndarray_int,
        np_ndarray_intp as np_ndarray_intp,
        np_ndarray_num as np_ndarray_num,
        np_ndarray_str as np_ndarray_str,
        np_ndarray_td as np_ndarray_td,
    )
else:
    # Builtin bool type and its string alias
    BuiltinBooleanDtypeArg: TypeAlias = type[bool] | Literal["bool"]
    # Pandas nullable boolean type and its string alias
    PandasBooleanDtypeArg: TypeAlias = pd.BooleanDtype | Literal["boolean"]
    # Numpy bool type
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_
    NumpyBooleanDtypeArg: TypeAlias = type[np.bool_] | Literal["?", "b1", "bool_"]
    # PyArrow boolean type and its string alias
    PyArrowBooleanDtypeArg: TypeAlias = Literal["bool[pyarrow]", "boolean[pyarrow]"]
    BooleanDtypeArg: TypeAlias = (
        BuiltinBooleanDtypeArg
        | PandasBooleanDtypeArg
        | NumpyBooleanDtypeArg
        | PyArrowBooleanDtypeArg
    )
    # Builtin integer type and its string alias
    BuiltinIntDtypeArg: TypeAlias = type[int] | Literal["int"]
    # Pandas nullable integer types and their string aliases
    PandasIntDtypeArg: TypeAlias = (
        pd.Int8Dtype
        | pd.Int16Dtype
        | pd.Int32Dtype
        | pd.Int64Dtype
        | Literal["Int8", "Int16", "Int32", "Int64"]
    )
    # Numpy signed integer types and their string aliases
    NumpyIntDtypeArg: TypeAlias = (
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.byte
        type[np.byte]  # noqa: PYI030
        | Literal["b", "i1", "int8", "byte"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.short
        | type[np.short]
        | Literal["h", "i2", "int16", "short"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.intc
        | type[np.intc]
        | Literal["i", "i4", "int32", "intc"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int_
        | type[np.int_]
        | Literal["l", "i8", "int64", "int_", "long"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.longlong
        | type[np.longlong]
        | Literal["q", "longlong"]  # NOTE: int128 not assigned
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.intp
        | type[np.intp]  # signed pointer (=`intptr_t`, platform dependent)
        | Literal["p", "intp"]
    )
    # PyArrow integer types and their string aliases
    PyArrowIntDtypeArg: TypeAlias = Literal[
        "int8[pyarrow]", "int16[pyarrow]", "int32[pyarrow]", "int64[pyarrow]"
    ]
    IntDtypeArg: TypeAlias = (
        BuiltinIntDtypeArg | PandasIntDtypeArg | NumpyIntDtypeArg | PyArrowIntDtypeArg
    )
    # Pandas nullable unsigned integer types and their string aliases
    PandasUIntDtypeArg: TypeAlias = (
        pd.UInt8Dtype
        | pd.UInt16Dtype
        | pd.UInt32Dtype
        | pd.UInt64Dtype
        | Literal["UInt8", "UInt16", "UInt32", "UInt64"]
    )
    # Numpy unsigned integer types and their string aliases
    NumpyUIntDtypeArg: TypeAlias = (
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ubyte
        type[np.ubyte]  # noqa: PYI030
        | Literal["B", "u1", "uint8", "ubyte"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ushort
        | type[np.ushort]
        | Literal["H", "u2", "uint16", "ushort"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintc
        | type[np.uintc]
        | Literal["I", "u4", "uint32", "uintc"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint
        | type[np.uint]
        | Literal["L", "u8", "uint", "ulong", "uint64"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.ulonglong
        | type[np.ulonglong]
        | Literal["Q", "ulonglong"]  # NOTE: uint128 not assigned
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uintp
        | type[np.uintp]  # unsigned pointer (=`uintptr_t`, platform dependent)
        | Literal["P", "uintp"]
    )
    # PyArrow unsigned integer types and their string aliases
    PyArrowUIntDtypeArg: TypeAlias = Literal[
        "uint8[pyarrow]", "uint16[pyarrow]", "uint32[pyarrow]", "uint64[pyarrow]"
    ]
    UIntDtypeArg: TypeAlias = (
        PandasUIntDtypeArg | NumpyUIntDtypeArg | PyArrowUIntDtypeArg
    )
    # Builtin float type and its string alias
    BuiltinFloatDtypeArg: TypeAlias = type[float] | Literal["float"]
    # Pandas nullable float types and their string aliases
    PandasFloatDtypeArg: TypeAlias = (
        pd.Float32Dtype | pd.Float64Dtype | Literal["Float32", "Float64"]
    )
    PandasAstypeFloatDtypeArg: TypeAlias = Literal["float_", "longfloat"]
    # Numpy float types and their string aliases
    NumpyFloat16DtypeArg: TypeAlias = (
        # NOTE: Alias np.float16 only on Linux x86_64, use np.half instead
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.half
        type[np.half]
        | Literal["e", "f2", "<f2", "float16", "half"]
    )
    NumpyFloatNot16DtypeArg: TypeAlias = (
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.single
        type[np.single]  # noqa: PYI030
        | Literal["f", "f4", "float32", "single"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double
        | type[np.double]
        | Literal["d", "f8", "float64", "double"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.longdouble
        | type[np.longdouble]
        | Literal["g", "f16", "float128", "longdouble"]
    )
    # PyArrow floating point types and their string aliases
    PyArrowFloatDtypeArg: TypeAlias = Literal[
        "float[pyarrow]",
        "double[pyarrow]",
        "float16[pyarrow]",
        "float32[pyarrow]",
        "float64[pyarrow]",
    ]
    FloatNotNumpy16DtypeArg: TypeAlias = (
        BuiltinFloatDtypeArg
        | PandasFloatDtypeArg
        | NumpyFloatNot16DtypeArg
        | PyArrowFloatDtypeArg
    )
    FloatDtypeArg: TypeAlias = (
        FloatNotNumpy16DtypeArg
        | NumpyFloat16DtypeArg
        | NumpyFloatNot16DtypeArg
        | PyArrowFloatDtypeArg
    )
    # Builtin complex type and its string alias
    BuiltinComplexDtypeArg: TypeAlias = type[complex] | Literal["complex"]
    PandasAstypeComplexDtypeArg: TypeAlias = (
        Literal["singlecomplex"]  # noqa: PYI030
        | Literal["cfloat", "complex_"]
        | Literal["c32", "complex256", "clongfloat", "longcomplex"]
    )
    # Numpy complex types and their aliases
    NumpyComplexDtypeArg: TypeAlias = (
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.csingle
        type[np.csingle]  # noqa: PYI030
        | Literal["F", "c8", "complex64", "csingle"]
        # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.cdouble
        | type[np.cdouble]
        | Literal["D", "c16", "complex128", "cdouble"]
        #  https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.clongdouble
        # NOTE: Alias np.complex256 only on Linux x86_64, use np.clongdouble instead
        | type[np.clongdouble]
        | Literal["G", "clongdouble"]
    )
    ComplexDtypeArg: TypeAlias = BuiltinComplexDtypeArg | NumpyComplexDtypeArg
    PandasAstypeTimedeltaDtypeArg: TypeAlias = Literal[
        "timedelta64[Y]",
        "timedelta64[M]",
        "timedelta64[W]",
        "timedelta64[D]",
        "timedelta64[h]",
        "timedelta64[m]",
        "timedelta64[μs]",
        "timedelta64[ps]",
        "timedelta64[fs]",
        "timedelta64[as]",
        # numpy type codes
        "m8[Y]",
        "m8[M]",
        "m8[W]",
        "m8[D]",
        "m8[h]",
        "m8[m]",
        "m8[μs]",
        "m8[ps]",
        "m8[fs]",
        "m8[as]",
        # little endian
        "<m8[Y]",
        "<m8[M]",
        "<m8[W]",
        "<m8[D]",
        "<m8[h]",
        "<m8[m]",
        "<m8[μs]",
        "<m8[ps]",
        "<m8[fs]",
        "<m8[as]",
    ]
    # Refer to https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
    NumpyTimedeltaDtypeArg: TypeAlias = Literal[
        "timedelta64[s]",
        "timedelta64[ms]",
        "timedelta64[us]",
        "timedelta64[ns]",
        # numpy type codes
        "m8[s]",
        "m8[ms]",
        "m8[us]",
        "m8[ns]",
        # little endian
        "<m8[s]",
        "<m8[ms]",
        "<m8[us]",
        "<m8[ns]",
    ]
    # PyArrow duration type and its string alias
    PyArrowTimedeltaDtypeArg: TypeAlias = Literal[
        "duration[s][pyarrow]",
        "duration[ms][pyarrow]",
        "duration[us][pyarrow]",
        "duration[ns][pyarrow]",
    ]
    TimedeltaDtypeArg: TypeAlias = NumpyTimedeltaDtypeArg | PyArrowTimedeltaDtypeArg
    # Pandas timestamp type and its string alias
    # Not comprehensive
    PandasTimestampDtypeArg: TypeAlias = (
        pd.DatetimeTZDtype
        | Literal[
            "datetime64[s, UTC]",
            "datetime64[ms, UTC]",
            "datetime64[us, UTC]",
            "datetime64[ns, UTC]",
        ]
    )
    PandasAstypeTimestampDtypeArg: TypeAlias = Literal[
        # numpy datetime64
        "datetime64[Y]",
        "datetime64[M]",
        "datetime64[W]",
        "datetime64[D]",
        "datetime64[h]",
        "datetime64[m]",
        "datetime64[μs]",
        "datetime64[ps]",
        "datetime64[fs]",
        "datetime64[as]",
        # numpy datetime64 type codes
        "M8[Y]",
        "M8[M]",
        "M8[W]",
        "M8[D]",
        "M8[h]",
        "M8[m]",
        "M8[μs]",
        "M8[ps]",
        "M8[fs]",
        "M8[as]",
        # little endian
        "<M8[Y]",
        "<M8[M]",
        "<M8[W]",
        "<M8[D]",
        "<M8[h]",
        "<M8[m]",
        "<M8[μs]",
        "<M8[ps]",
        "<M8[fs]",
        "<M8[as]",
    ]
    # Numpy timestamp type and its string alias
    NumpyTimestampDtypeArg: TypeAlias = Literal[
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        # numpy type codes
        "M8[s]",
        "M8[ms]",
        "M8[us]",
        "M8[ns]",
        # little endian
        "<M8[s]",
        "<M8[ms]",
        "<M8[us]",
        "<M8[ns]",
    ]
    # PyArrow timestamp type and its string alias
    PyArrowTimestampDtypeArg: TypeAlias = Literal[
        "date32[pyarrow]",
        "date64[pyarrow]",
        "timestamp[s][pyarrow]",
        "timestamp[ms][pyarrow]",
        "timestamp[us][pyarrow]",
        "timestamp[ns][pyarrow]",
    ]
    TimestampDtypeArg: TypeAlias = (
        PandasTimestampDtypeArg | NumpyTimestampDtypeArg | PyArrowTimestampDtypeArg
    )
    # Builtin str type and its string alias
    BuiltinStrDtypeArg: TypeAlias = type[str] | Literal["str"]
    # Pandas nullable string type and its string alias
    PandasStrDtypeArg: TypeAlias = pd.StringDtype | Literal["string"]
    # Numpy string type and its string alias
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.str_
    NumpyStrDtypeArg: TypeAlias = type[np.str_] | Literal["U", "str_", "unicode"]
    # PyArrow string type and its string alias
    PyArrowStrDtypeArg: TypeAlias = Literal["string[pyarrow]"]
    StrDtypeArg: TypeAlias = (
        BuiltinStrDtypeArg | PandasStrDtypeArg | NumpyStrDtypeArg | PyArrowStrDtypeArg
    )
    # Builtin bytes type and its string alias
    BuiltinBytesDtypeArg: TypeAlias = type[bytes] | Literal["bytes"]
    # Numpy bytes type and its string alias
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bytes_
    NumpyBytesDtypeArg: TypeAlias = type[np.bytes_] | Literal["S", "bytes_"]
    # PyArrow binary type and its string alias
    PyArrowBytesDtypeArg: TypeAlias = Literal["binary[pyarrow]"]
    BytesDtypeArg: TypeAlias = (
        BuiltinBytesDtypeArg | NumpyBytesDtypeArg | PyArrowBytesDtypeArg
    )
    CategoryDtypeArg: TypeAlias = pd.CategoricalDtype | Literal["category"]

    # Builtin object type and its string alias
    BuiltinObjectDtypeArg: TypeAlias = type[object] | Literal["object"]
    # Numpy object type and its string alias
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.object_
    # NOTE: "object_" not assigned
    NumpyObjectDtypeArg: TypeAlias = type[np.object_] | Literal["O"]

    ObjectDtypeArg: TypeAlias = BuiltinObjectDtypeArg | NumpyObjectDtypeArg

    # Numpy void type and its string alias
    # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.void
    NumpyVoidDtypeArg: TypeAlias = type[np.void] | Literal["V", "void"]
    VoidDtypeArg: TypeAlias = NumpyVoidDtypeArg

    BuiltinDtypeArg: TypeAlias = (
        BuiltinBooleanDtypeArg
        | BuiltinIntDtypeArg
        | BuiltinFloatDtypeArg
        | BuiltinComplexDtypeArg
        | BuiltinStrDtypeArg
        | BuiltinBytesDtypeArg
        | BuiltinObjectDtypeArg
    )
    NumpyNotTimeDtypeArg: TypeAlias = (
        NumpyBooleanDtypeArg
        | NumpyIntDtypeArg
        | NumpyUIntDtypeArg
        | NumpyFloat16DtypeArg
        | NumpyFloatNot16DtypeArg
        | NumpyComplexDtypeArg
        | NumpyStrDtypeArg
        | NumpyBytesDtypeArg
        | NumpyObjectDtypeArg
        | NumpyVoidDtypeArg
    )
    PyArrowNotStrDtypeArg: TypeAlias = (
        PyArrowBooleanDtypeArg
        | PyArrowIntDtypeArg
        | PyArrowUIntDtypeArg
        | PyArrowFloatDtypeArg
        | PyArrowTimedeltaDtypeArg
        | PyArrowTimestampDtypeArg
        | PyArrowBytesDtypeArg
    )

    _G = TypeVar("_G", bound=np.generic)
    _S = TypeVar("_S", bound=tuple[int, ...])
    # Separately define here so pytest works
    np_1darray: TypeAlias = np.ndarray[tuple[int], np.dtype[_G]]
    np_1darray_bool: TypeAlias = np.ndarray[tuple[int], np.bool_]
    np_1darray_str: TypeAlias = np.ndarray[tuple[int], np.str_]
    np_1darray_bytes: TypeAlias = np.ndarray[tuple[int], np.bytes_]
    np_1darray_complex: TypeAlias = np.ndarray[tuple[int], np.complexfloating]
    np_1darray_object: TypeAlias = np.ndarray[tuple[int], np.object_]
    np_1darray_intp: TypeAlias = np_1darray[np.intp]
    np_1darray_int64: TypeAlias = np_1darray[np.int64]
    np_1darray_anyint: TypeAlias = np_1darray[np.integer]
    np_1darray_float: TypeAlias = np_1darray[np.floating]
    np_1darray_dt: TypeAlias = np_1darray[np.datetime64]
    np_1darray_td: TypeAlias = np_1darray[np.timedelta64]
    np_2darray: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_G]]
    np_ndarray: TypeAlias = np.ndarray[_S, np.dtype[_G]]
    np_ndarray_bool: TypeAlias = npt.NDArray[np.bool_]
    np_ndarray_dt: TypeAlias = npt.NDArray[np.datetime64]
    np_ndarray_int: TypeAlias = npt.NDArray[np.signedinteger]
    np_ndarray_intp: TypeAlias = npt.NDArray[np.intp]
    np_ndarray_int64: TypeAlias = npt.NDArray[np.int64]
    np_ndarray_num: TypeAlias = npt.NDArray[np.number]
    np_ndarray_str: TypeAlias = npt.NDArray[np.str_]
    np_ndarray_td: TypeAlias = npt.NDArray[np.timedelta64]

TYPE_CHECKING_INVALID_USAGE: Final = TYPE_CHECKING
LINUX = sys.platform == "linux"
WINDOWS = sys.platform in {"win32", "cygwin"}
MAC = sys.platform == "darwin"
PD_LTE_23 = Version(pd.__version__) < Version("2.3.999")
NUMPY20 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"

NATIVE_FLOAT_ARGS = {float: np.floating, "float": np.floating}
NUMPY_FLOAT16_ARGS = {
    np.half: np.half,
    "half": np.half,
    "e": np.half,
    "float16": np.float16,
    "f2": np.float16,
}
NUMPY_FLOAT_NOT16_ARGS = {
    # numpy float32
    np.single: np.single,
    "single": np.single,
    "f": np.single,
    "float32": np.float32,
    "f4": np.float32,
    # numpy float64
    np.double: np.double,
    "double": np.double,
    "d": np.double,
    "float64": np.float64,
    "f8": np.float64,
    # numpy float128
    np.longdouble: np.longdouble,
    "g": np.longdouble,
}
PYARROW_FLOAT_ARGS = {
    # pyarrow float32
    "float32[pyarrow]": float,
    "float[pyarrow]": float,
    # pyarrow float64
    "float64[pyarrow]": float,
    "double[pyarrow]": float,
}
PANDAS_FLOAT_ARGS = {
    # pandas Float32
    pd.Float32Dtype(): np.float32,
    "Float32": np.float32,
    # pandas Float64
    pd.Float64Dtype(): np.float64,
    "Float64": np.float64,
}
TYPE_FLOAT_NOT_NUMPY16_ARGS = (
    NATIVE_FLOAT_ARGS | NUMPY_FLOAT_NOT16_ARGS | PYARROW_FLOAT_ARGS | PANDAS_FLOAT_ARGS
)
TYPE_FLOAT_ARGS = TYPE_FLOAT_NOT_NUMPY16_ARGS | NUMPY_FLOAT16_ARGS
ASTYPE_FLOAT_NOT_NUMPY16_ARGS = {
    **TYPE_FLOAT_NOT_NUMPY16_ARGS,
    "longdouble": np.longdouble,
    "f16": np.longdouble,
    # "float96": np.longdouble,  # NOTE: unsupported
    "float128": np.longdouble,  # NOTE: UNIX ONLY
}
ASTYPE_FLOAT_ARGS = ASTYPE_FLOAT_NOT_NUMPY16_ARGS | NUMPY_FLOAT16_ARGS


def check(
    actual: T,
    klass: type,
    dtype: type | None = None,
    attr: str = "left",
    index_to_check_for_type: Literal[0, -1] = 0,
) -> T:
    __tracebackhide__ = True
    origin = get_origin(klass)
    if not isinstance(actual, origin or klass):
        raise RuntimeError(f"Expected type '{klass}' but got '{type(actual)}'")
    if origin is np.ndarray:
        # Check shape and dtype
        args = get_args(klass)
        shape_type = args[0] if len(args) >= 1 else None
        dtype_type = args[1] if len(args) >= 2 else None
        if (
            shape_type
            and get_origin(shape_type) is tuple
            and (tuple_args := get_args(shape_type))
            and ... not in tuple_args  # fixed-length tuple
            and (arr_ndim := getattr(actual, "ndim"))  # noqa: B009
            != (expected_ndim := len(tuple_args))
        ):
            raise RuntimeError(
                f"Array has wrong dimension {arr_ndim}, expected {expected_ndim}"
            )

        if (
            dtype_type
            and get_origin(dtype_type) is np.dtype
            and (dtype_args := get_args(dtype_type))
            and isinstance((expected_dtype := dtype_args[0]), type)
            and issubclass(expected_dtype, np.generic)
            and (arr_dtype := getattr(actual, "dtype")) != expected_dtype  # noqa: B009
        ):
            raise RuntimeError(
                f"Array has wrong dtype {arr_dtype}, expected {expected_dtype.__name__}"
            )

    if dtype is None:
        return actual

    if isinstance(actual, pd.Series):
        value = actual.iloc[index_to_check_for_type]
    elif isinstance(actual, pd.Index):
        value = actual[index_to_check_for_type]
    elif isinstance(actual, BaseGroupBy):
        value = actual.obj
    elif hasattr(actual, "__iter__"):
        value = next(
            iter(actual)  # pyright: ignore[reportArgumentType,reportCallIssue]
        )
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)

    if not isinstance(value, dtype):
        raise RuntimeError(f"Expected type '{dtype}' but got '{type(value)}'")
    return actual


def pytest_warns_bounded(
    warning: type[Warning],
    match: str,
    lower: str | None = None,
    upper: str | None = None,
    version_str: str | None = None,
    upper_exception: type[Exception] | None = None,
) -> AbstractContextManager:
    """
    Version conditional pytest.warns context manager

    Returns a context manager that will raise an error if
    the warning is not issued when pandas version is
    between the lower and upper version given.

    Parameters
    ----------
    warning : type[Warning]
        The warning class to check for.
    match : str
        The string to match in the warning message.
    lower : str, optional
        The lower bound of the version to check for the warning.
    upper : str, optional
        The upper bound of the version to check for the warning.
    version_str: str, optional
        The version string to use.  If None, then uses the pandas version.
        Can be used to check a python version as well
    upper_exception: Exception, optional
        Exception to catch if the pandas version is greater than or equal to
        the upper bound

    Notes
    -----
    The lower and upper bounds are exclusive so that a pytest.warns context
    manager is returned if lower < version_str < upper.

    Examples
    --------
    with pytest_warns_bounded(UserWarning, match="foo", lower="1.2.99"):
        # Versions 1.3.0 and above will raise an error
        # if the warning is not issued
        pass

    with pytest_warns_bounded(UserWarning, match="foo", upper="1.5.99"):
        # Versions 1.6.0 and below will raise an error
        # if the warning is not issued
        pass

    with pytest_warns_bounded(
        UserWarning, match="foo", lower="1.2.99", upper="1.5.99"
    ):
        # Versions between 1.3.x and 1.5.x will raise an error
        pass

    with pytest_warns_bounded(
        UserWarning, match="foo", lower="3.10",
        version_str = platform.python_version()
    ):
        # Python version 3.11 and above will raise an error
        # if the warning is not issued
        pass

    with pytest_warns_bounded(
        UserWarning, match="foo", lower="1.2.99", upper="1.5.99",
        upper_exception=AttributeError
    ):
        # Versions between 1.3.x and 1.5.x will raise an error
        # Above 1.5.x, we expect an `AttributeError` to be raised
        pass

    """
    lb = Version("0.0.0") if lower is None else Version(lower)
    ub = Version("9999.0.0") if upper is None else Version(upper)
    if version_str is None:
        current = Version(pd.__version__)
    else:
        current = Version(version_str)
    if lb < current < ub:
        return pytest.warns(warning, match=match)
    if upper_exception is None:
        return nullcontext()
    return suppress(upper_exception)


def exception_on_platform(dtype: type | str | ExtensionDtype) -> type[Exception] | None:
    if (WINDOWS or MAC) and dtype in {"f16", "float128"}:
        return TypeError
    return None
