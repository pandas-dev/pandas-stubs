from datetime import (
    date,
    datetime,
    timedelta,
)

import numpy as np
import pandas as pd

from tests import LINUX
from tests._typing import (
    BooleanDtypeArg,
    BuiltinBooleanDtypeArg,
    BuiltinBytesDtypeArg,
    BuiltinComplexDtypeArg,
    BuiltinDtypeArg,
    BuiltinFloatDtypeArg,
    BuiltinIntDtypeArg,
    BuiltinObjectDtypeArg,
    BuiltinStrDtypeArg,
    BytesDtypeArg,
    CategoryDtypeArg,
    ComplexDtypeArg,
    FloatDtypeArg,
    IntDtypeArg,
    NumpyBooleanDtypeArg,
    NumpyBytesDtypeArg,
    NumpyComplexDtypeArg,
    NumpyFloat16DtypeArg,
    NumpyFloatNot16DtypeArg,
    NumpyIntDtypeArg,
    NumpyNotTimeDtypeArg,
    NumpyObjectDtypeArg,
    NumpyStrDtypeArg,
    NumpyTimedeltaDtypeArg,
    NumpyTimestampDtypeArg,
    NumpyUIntDtypeArg,
    NumpyVoidDtypeArg,
    ObjectDtypeArg,
    PandasAstypeTimedeltaDtypeArg,
    PandasAstypeTimestampDtypeArg,
    PandasBaseStrDtypeArg,
    PandasBooleanDtypeArg,
    PandasFloatDtypeArg,
    PandasIntDtypeArg,
    PandasStrDtypeArg,
    PandasTimestampDtypeArg,
    PandasUIntDtypeArg,
    PyArrowBooleanDtypeArg,
    PyArrowBytesDtypeArg,
    PyArrowFloatDtypeArg,
    PyArrowIntDtypeArg,
    PyArrowStrDtypeArg,
    PyArrowTimedeltaDtypeArg,
    PyArrowTimestampDtypeArg,
    PyArrowUIntDtypeArg,
    StrDtypeArg,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    UIntDtypeArg,
    VoidDtypeArg,
)

PYTHON_BOOL_ARGS = dict.fromkeys((bool, "bool"), np.bool_)
PANDAS_BOOL_ARGS = dict.fromkeys((pd.BooleanDtype(), "boolean"), np.bool_)
NUMPY_BOOL_ARGS = dict.fromkeys((np.bool_, "?", "b1", "bool_"), np.bool_)
PYARROW_BOOL_ARGS = dict.fromkeys(("bool[pyarrow]", "boolean[pyarrow]"), bool)
ASTYPE_BOOL_ARGS = (
    PYTHON_BOOL_ARGS | PANDAS_BOOL_ARGS | NUMPY_BOOL_ARGS | PYARROW_BOOL_ARGS
)

PYTHON_INT_ARGS = dict.fromkeys((int, "int"), np.integer)
PANDAS_INT_ARGS = {
    **dict.fromkeys((pd.Int8Dtype(), "Int8"), np.int8),  # pandas Int8
    **dict.fromkeys((pd.Int16Dtype(), "Int16"), np.int16),  # pandas Int16
    **dict.fromkeys((pd.Int32Dtype(), "Int32"), np.int32),  # pandas Int32
    **dict.fromkeys((pd.Int64Dtype(), "Int64"), np.int64),  # pandas Int64
}
NUMPY_INT_ARGS: dict[str | type[np.signedinteger], type[np.signedinteger]] = {
    # numpy int8
    **dict.fromkeys((np.byte, "byte", "b"), np.byte),
    **dict.fromkeys(("int8", "i1"), np.int8),
    # numpy int16
    **dict.fromkeys((np.short, "short", "h"), np.short),
    **dict.fromkeys(("int16", "i2"), np.int16),
    # numpy int32
    **dict.fromkeys((np.intc, "intc", "i"), np.intc),
    **dict.fromkeys(("int32", "i4"), np.int32),
    # numpy long
    **dict.fromkeys(("long", "l"), np.long),
    # numpy int64
    **dict.fromkeys((np.int_, "int_"), np.int_),
    **dict.fromkeys(("int64", "i8"), np.int64),
    # numpy extended int
    **dict.fromkeys((np.longlong, "longlong", "q"), np.longlong),
    # numpy signed pointer  (platform dependent one of int[8,16,32,64])
    **dict.fromkeys((np.intp, "intp", "p"), np.intp),
}
PYARROW_INT_ARGS = dict.fromkeys([f"int{b}[pyarrow]" for b in (8, 16, 32, 64)], int)
ASTYPE_INT_ARGS = PYTHON_INT_ARGS | PANDAS_INT_ARGS | NUMPY_INT_ARGS | PYARROW_INT_ARGS

PANDAS_UINT_ARGS = {
    **dict.fromkeys((pd.UInt8Dtype(), "UInt8"), np.uint8),  # pandas UInt8
    **dict.fromkeys((pd.UInt16Dtype(), "UInt16"), np.uint16),  # pandas UInt16
    **dict.fromkeys((pd.UInt32Dtype(), "UInt32"), np.uint32),  # pandas UInt32
    **dict.fromkeys((pd.UInt64Dtype(), "UInt64"), np.uint64),  # pandas UInt64
}
NUMPY_UINT_ARGS: dict[str | type[np.unsignedinteger], type[np.unsignedinteger]] = {
    # numpy uint8
    **dict.fromkeys((np.ubyte, "ubyte", "B"), np.ubyte),
    **dict.fromkeys(("uint8", "u1"), np.uint8),
    # numpy uint16
    **dict.fromkeys((np.ushort, "ushort", "H"), np.ushort),
    **dict.fromkeys(("uint16", "u2"), np.uint16),
    # numpy uint32
    **dict.fromkeys((np.uintc, "uintc", "I"), np.uintc),
    **dict.fromkeys(("uint32", "u4"), np.uint32),
    # numpy ulong
    **dict.fromkeys(("ulong", "L"), np.ulong),
    # numpy uint64
    **dict.fromkeys((np.uint, "uint"), np.uint),
    **dict.fromkeys(("uint64", "u8"), np.uint64),
    # numpy extended uint
    **dict.fromkeys((np.ulonglong, "ulonglong", "Q"), np.ulonglong),
    # numpy unsigned pointer  (platform dependent one of uint[8,16,32,64])
    **dict.fromkeys((np.uintp, "uintp", "P"), np.uintp),
}
PYARROW_UINT_ARGS = dict.fromkeys([f"uint{b}[pyarrow]" for b in (8, 16, 32, 64)], int)
ASTYPE_UINT_ARGS = PANDAS_UINT_ARGS | NUMPY_UINT_ARGS | PYARROW_UINT_ARGS

PYTHON_FLOAT_ARGS = dict.fromkeys((float, "float"), np.floating)
PANDAS_FLOAT_ARGS = {
    **dict.fromkeys((pd.Float32Dtype(), "Float32"), np.float32),  # pandas Float32
    **dict.fromkeys((pd.Float64Dtype(), "Float64"), np.float64),  # pandas Float64
}
NUMPY_FLOAT16_ARGS: dict[str | type[np.floating], type[np.floating]] = {
    **dict.fromkeys((np.half, "half", "e"), np.half),
    **dict.fromkeys(("float16", "f2", "<f2"), np.float16),
}
NUMPY_FLOAT_NOT16_ARGS: dict[str | type[np.floating], type[np.floating]] = {
    # numpy float32
    **dict.fromkeys((np.single, "single", "f"), np.single),
    **dict.fromkeys(("float32", "f4"), np.float32),
    # numpy float64
    **dict.fromkeys((np.double, "double", "d"), np.double),
    **dict.fromkeys(("float64", "f8"), np.float64),
    # numpy float128
    **dict.fromkeys((np.longdouble, "longdouble", "g"), np.longdouble),
    **dict.fromkeys(("f16", "float128"), np.float128 if LINUX else np.longdouble),
    # "float96": np.longdouble,  # NOTE: unsupported
}
PYARROW_FLOAT_ARGS = {
    "float16[pyarrow]": float,  # pyarrow float16
    **dict.fromkeys(("float32[pyarrow]", "float[pyarrow]"), float),  # pyarrow float32
    **dict.fromkeys(("float64[pyarrow]", "double[pyarrow]"), float),  # pyarrow float64
}
ASTYPE_FLOAT_NOT_NUMPY16_ARGS = (
    PYTHON_FLOAT_ARGS | PANDAS_FLOAT_ARGS | NUMPY_FLOAT_NOT16_ARGS | PYARROW_FLOAT_ARGS
)
ASTYPE_FLOAT_ARGS = (
    (PYTHON_FLOAT_ARGS | PANDAS_FLOAT_ARGS)
    | (NUMPY_FLOAT16_ARGS | NUMPY_FLOAT_NOT16_ARGS)
    | PYARROW_FLOAT_ARGS
)

PYTHON_COMPLEX_ARGS = dict.fromkeys((complex, "complex"), np.complexfloating)
NUMPY_COMPLEX_ARGS: dict[str | type[np.complexfloating], type[np.complexfloating]] = {
    # numpy complex64
    **dict.fromkeys((np.csingle, "csingle", "F"), np.csingle),
    **dict.fromkeys(("complex64", "c8"), np.complex64),
    # numpy complex128
    **dict.fromkeys((np.cdouble, "cdouble", "D"), np.cdouble),
    **dict.fromkeys(("complex128", "c16"), np.complex128),
    # numpy complex256
    **dict.fromkeys((np.clongdouble, "clongdouble", "G"), np.clongdouble),
    **dict.fromkeys(("c32", "complex256"), np.complex256 if LINUX else np.clongdouble),
    # "complex192": np.clongdouble,  # NOTE: unsupported
}
ASTYPE_COMPLEX_ARGS = PYTHON_COMPLEX_ARGS | NUMPY_COMPLEX_ARGS

NUMPY_UNITS = ("s", "ms", "us", "ns")
PANDAS_UNITS = ("Y", "M", "W", "D", "h", "m", "μs", "ps", "fs", "as")

NUMPY_TIMESTAMP_ARGS = {
    # numpy datetime64
    **dict.fromkeys([f"datetime64[{u}]" for u in NUMPY_UNITS], datetime),
    # numpy datetime64 type codes
    **dict.fromkeys([f"M8[{u}]" for u in NUMPY_UNITS], datetime),
    # little endian
    **dict.fromkeys([f"<M8[{u}]" for u in NUMPY_UNITS], datetime),
}
PANDAS_TIMESTAMP_ARGS = dict.fromkeys(
    [f"datetime64[{u}, UTC]" for u in NUMPY_UNITS], datetime
)
PANDAS_ASTYPE_TIMESTAMP_ARGS = {
    # numpy datetime64
    **dict.fromkeys([f"datetime64[{u}]" for u in PANDAS_UNITS], datetime),
    # numpy datetime64 type codes
    **dict.fromkeys([f"M8[{u}]" for u in PANDAS_UNITS], datetime),
    # little endian
    **dict.fromkeys([f"<M8[{u}]" for u in PANDAS_UNITS], datetime),
}
PYARROW_TIMESTAMP_ARGS = {
    # pyarrow timestamp
    **dict.fromkeys([f"timestamp[{u}][pyarrow]" for u in NUMPY_UNITS], datetime),
    # pyarrow date
    **dict.fromkeys(("date32[pyarrow]", "date64[pyarrow]"), date),
}
TYPE_TIMESTAMP_ARGS = (
    NUMPY_TIMESTAMP_ARGS | PANDAS_TIMESTAMP_ARGS | PYARROW_TIMESTAMP_ARGS
)
ASTYPE_TIMESTAMP_ARGS = TYPE_TIMESTAMP_ARGS | PANDAS_ASTYPE_TIMESTAMP_ARGS

NUMPY_TIMEDELTA_ARGS = {
    # numpy timedelta64
    **dict.fromkeys([f"timedelta64[{u}]" for u in NUMPY_UNITS], timedelta),
    # numpy timedelta64 type codes
    **dict.fromkeys([f"m8[{u}]" for u in NUMPY_UNITS], timedelta),
    # little endian
    **dict.fromkeys([f"<m8[{u}]" for u in NUMPY_UNITS], timedelta),
}
PANDAS_ASTYPE_TIMEDELTA_ARGS = {
    # numpy timedelta64
    **dict.fromkeys([f"timedelta64[{u}]" for u in PANDAS_UNITS], timedelta),
    # numpy timedelta64 type codes
    **dict.fromkeys([f"m8[{u}]" for u in PANDAS_UNITS], timedelta),
    # little endian
    **dict.fromkeys([f"<m8[{u}]" for u in PANDAS_UNITS], timedelta),
}
# pyarrow duration
PYARROW_TIMEDELTA_ARGS = dict.fromkeys(
    [f"duration[{u}][pyarrow]" for u in NUMPY_UNITS], timedelta
)
TYPE_TIMEDELTA_ARGS = NUMPY_TIMEDELTA_ARGS | PYARROW_TIMEDELTA_ARGS
ASTYPE_TIMEDELTA_ARGS = TYPE_TIMEDELTA_ARGS | PANDAS_ASTYPE_TIMEDELTA_ARGS

PYTHON_STRING_ARGS = dict.fromkeys((str, "str"), str)
PANDAS_BASE_STRING_ARGS = dict.fromkeys((pd.StringDtype(), "string"), str)
PANDAS_STRING_ARGS = dict.fromkeys((pd.StringDtype("python"), "string[python]"), str)
NUMPY_STRING_ARGS = dict.fromkeys((np.str_, "str_", "unicode", "U"), str)
PYARROW_STRING_ARGS = dict.fromkeys((pd.StringDtype("pyarrow"), "string[pyarrow]"), str)
ASTYPE_STRING_ARGS = (
    PYTHON_STRING_ARGS
    | PANDAS_BASE_STRING_ARGS
    | PANDAS_STRING_ARGS
    | PYARROW_STRING_ARGS
    | NUMPY_STRING_ARGS
    | PYARROW_STRING_ARGS
)

PYTHON_BYTES_ARGS = dict.fromkeys((bytes, "bytes"), bytes)
NUMPY_BYTES_ARGS = dict.fromkeys((np.bytes_, "S", "bytes_"), np.bytes_)
PYARROW_BYTES_ARGS = {"binary[pyarrow]": bytes}
ASTYPE_BYTES_ARGS = PYTHON_BYTES_ARGS | NUMPY_BYTES_ARGS | PYARROW_BYTES_ARGS

ASTYPE_CATEGORICAL_ARGS = {
    # pandas category
    **dict.fromkeys((pd.CategoricalDtype(), "category"), object),
    # pyarrow dictionary
    # ("dictionary[pyarrow]", "pd.Series[category]", Categorical),
}

PYTHON_OBJECT_ARGS = dict.fromkeys((object, "object"), object)
NUMPY_OBJECT_ARGS = dict.fromkeys((np.object_, "object_", "O"), object)
ASTYPE_OBJECT_ARGS = PYTHON_OBJECT_ARGS | NUMPY_OBJECT_ARGS

NUMPY_VOID_ARGS = dict.fromkeys((np.void, "void", "V"), np.void)
ASTYPE_VOID_ARGS = NUMPY_VOID_ARGS

PYTHON_DTYPE_ARGS = (
    PYTHON_BOOL_ARGS
    | PYTHON_INT_ARGS
    | PYTHON_FLOAT_ARGS
    | PYTHON_COMPLEX_ARGS
    | PYTHON_STRING_ARGS
    | PYTHON_BYTES_ARGS
    | PYTHON_OBJECT_ARGS
)
NUMPY_NOT_DATETIMELIKE_DTYPE_ARGS = (
    NUMPY_BOOL_ARGS
    | NUMPY_INT_ARGS
    | NUMPY_UINT_ARGS
    | NUMPY_FLOAT16_ARGS
    | NUMPY_FLOAT_NOT16_ARGS
    | NUMPY_COMPLEX_ARGS
    | NUMPY_STRING_ARGS
    | NUMPY_BYTES_ARGS
    | NUMPY_OBJECT_ARGS
    | NUMPY_VOID_ARGS
)


DTYPE_ARG_ALIAS_MAPS = {
    BooleanDtypeArg: ASTYPE_BOOL_ARGS,
    BuiltinBooleanDtypeArg: PYTHON_BOOL_ARGS,
    BuiltinBytesDtypeArg: PYTHON_BYTES_ARGS,
    BuiltinComplexDtypeArg: PYTHON_COMPLEX_ARGS,
    BuiltinDtypeArg: PYTHON_DTYPE_ARGS,
    BuiltinFloatDtypeArg: PYTHON_FLOAT_ARGS,
    BuiltinIntDtypeArg: PYTHON_INT_ARGS,
    BuiltinObjectDtypeArg: PYTHON_OBJECT_ARGS,
    BuiltinStrDtypeArg: PYTHON_STRING_ARGS,
    BytesDtypeArg: ASTYPE_BYTES_ARGS,
    CategoryDtypeArg: ASTYPE_CATEGORICAL_ARGS,
    ComplexDtypeArg: ASTYPE_COMPLEX_ARGS,
    FloatDtypeArg: ASTYPE_FLOAT_ARGS,
    IntDtypeArg: ASTYPE_INT_ARGS,
    NumpyBooleanDtypeArg: NUMPY_BOOL_ARGS,
    NumpyBytesDtypeArg: NUMPY_BYTES_ARGS,
    NumpyComplexDtypeArg: NUMPY_COMPLEX_ARGS,
    NumpyFloat16DtypeArg: NUMPY_FLOAT16_ARGS,
    NumpyFloatNot16DtypeArg: NUMPY_FLOAT_NOT16_ARGS,
    NumpyIntDtypeArg: NUMPY_INT_ARGS,
    NumpyNotTimeDtypeArg: NUMPY_NOT_DATETIMELIKE_DTYPE_ARGS,
    NumpyObjectDtypeArg: NUMPY_OBJECT_ARGS,
    NumpyStrDtypeArg: NUMPY_STRING_ARGS,
    NumpyTimedeltaDtypeArg: NUMPY_TIMEDELTA_ARGS,
    NumpyTimestampDtypeArg: NUMPY_TIMESTAMP_ARGS,
    NumpyUIntDtypeArg: NUMPY_UINT_ARGS,
    NumpyVoidDtypeArg: NUMPY_VOID_ARGS,
    ObjectDtypeArg: ASTYPE_OBJECT_ARGS,
    PandasAstypeTimedeltaDtypeArg: PANDAS_ASTYPE_TIMEDELTA_ARGS,
    PandasAstypeTimestampDtypeArg: PANDAS_ASTYPE_TIMESTAMP_ARGS,
    PandasBaseStrDtypeArg: PANDAS_BASE_STRING_ARGS,
    PandasBooleanDtypeArg: PANDAS_BOOL_ARGS,
    PandasFloatDtypeArg: PANDAS_FLOAT_ARGS,
    PandasIntDtypeArg: PANDAS_INT_ARGS,
    PandasStrDtypeArg: PANDAS_STRING_ARGS,
    PandasTimestampDtypeArg: PANDAS_TIMESTAMP_ARGS,
    PandasUIntDtypeArg: PANDAS_UINT_ARGS,
    PyArrowBooleanDtypeArg: PYARROW_BOOL_ARGS,
    PyArrowBytesDtypeArg: PYARROW_BYTES_ARGS,
    PyArrowFloatDtypeArg: PYARROW_FLOAT_ARGS,
    PyArrowIntDtypeArg: PYARROW_INT_ARGS,
    PyArrowStrDtypeArg: PYARROW_STRING_ARGS,
    PyArrowTimedeltaDtypeArg: PYARROW_TIMEDELTA_ARGS,
    PyArrowTimestampDtypeArg: PYARROW_TIMESTAMP_ARGS,
    PyArrowUIntDtypeArg: PYARROW_UINT_ARGS,
    StrDtypeArg: ASTYPE_STRING_ARGS,
    TimedeltaDtypeArg: TYPE_TIMEDELTA_ARGS,
    TimestampDtypeArg: TYPE_TIMESTAMP_ARGS,
    UIntDtypeArg: ASTYPE_UINT_ARGS,
    VoidDtypeArg: ASTYPE_VOID_ARGS,
}
