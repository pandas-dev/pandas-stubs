from __future__ import annotations

from collections.abc import Iterable
from contextlib import (
    AbstractContextManager,
    nullcontext,
    suppress,
)
from datetime import (
    date,
    datetime,
    timedelta,
)
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    cast,
    get_args,
    get_origin,
)

import numpy as np
import pandas as pd

# Next set of imports is to keep the private imports needed for testing
# in one place
from pandas._testing import ensure_clean as ensure_clean
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.util.version import Version
import pytest

from pandas.core.dtypes.base import ExtensionDtype

if TYPE_CHECKING:
    from pandas._typing import T

TYPE_CHECKING_INVALID_USAGE: Final = TYPE_CHECKING
LINUX = sys.platform == "linux"
WINDOWS = sys.platform in {"win32", "cygwin"}
MAC = sys.platform == "darwin"
PD_LTE_23 = Version(pd.__version__) < Version("2.3.999")
NUMPY20 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"

PYTHON_BOOL_ARGS = {bool: np.bool_, "bool": np.bool_}
PANDAS_BOOL_ARGS = {pd.BooleanDtype(): np.bool_, "boolean": np.bool_}
NUMPY_BOOL_ARGS = {np.bool_: np.bool_, "bool_": np.bool_, "?": np.bool_, "b1": np.bool_}
PYARROW_BOOL_ARGS = {"bool[pyarrow]": bool, "boolean[pyarrow]": bool}
ASTYPE_BOOL_ARGS = (
    PYTHON_BOOL_ARGS | PANDAS_BOOL_ARGS | NUMPY_BOOL_ARGS | PYARROW_BOOL_ARGS
)

PYTHON_INT_ARGS = {int: np.integer, "int": np.integer}
PANDAS_INT_ARGS = {
    # pandas Int8
    pd.Int8Dtype(): np.int8,
    "Int8": np.int8,
    # pandas Int16
    pd.Int16Dtype(): np.int16,
    "Int16": np.int16,
    # pandas Int32
    pd.Int32Dtype(): np.int32,
    "Int32": np.int32,
    # pandas Int64
    pd.Int64Dtype(): np.int64,
    "Int64": np.int64,
}
NUMPY_INT_ARGS = {
    # numpy int8
    np.byte: np.byte,
    "byte": np.byte,
    "b": np.byte,
    "int8": np.int8,
    "i1": np.int8,
    # numpy int16
    np.short: np.short,
    "short": np.short,
    "h": np.short,
    "int16": np.int16,
    "i2": np.int16,
    # numpy int32
    np.intc: np.intc,
    "intc": np.intc,
    "i": np.intc,
    "int32": np.int32,
    "i4": np.int32,
    # numpy int64
    np.int_: np.int_,
    "int_": np.int_,
    "int64": np.int64,
    "i8": np.int64,
    # numpy extended int
    np.longlong: np.longlong,
    "longlong": np.longlong,
    "q": np.longlong,
    # numpy signed pointer  (platform dependent one of int[8,16,32,64])
    np.intp: np.intp,
    "intp": np.intp,
    "p": np.intp,
}
PYARROW_INT_ARGS = {
    "int8[pyarrow]": int,
    "int16[pyarrow]": int,
    "int32[pyarrow]": int,
    "int64[pyarrow]": int,
}
ASTYPE_INT_ARGS = PYTHON_INT_ARGS | PANDAS_INT_ARGS | NUMPY_INT_ARGS | PYARROW_INT_ARGS

PANDAS_UINT_ARGS = {
    # pandas UInt8
    pd.UInt8Dtype(): np.uint8,
    "UInt8": np.uint8,
    # pandas UInt16
    pd.UInt16Dtype(): np.uint16,
    "UInt16": np.uint16,
    # pandas UInt32
    pd.UInt32Dtype(): np.uint32,
    "UInt32": np.uint32,
    # pandas UInt64
    pd.UInt64Dtype(): np.uint64,
    "UInt64": np.uint64,
}
NUMPY_UINT_ARGS = {
    # numpy uint8
    np.ubyte: np.ubyte,
    "ubyte": np.ubyte,
    "B": np.ubyte,
    "uint8": np.uint8,
    "u1": np.uint8,
    # numpy uint16
    np.ushort: np.ushort,
    "ushort": np.ushort,
    "H": np.ushort,
    "uint16": np.uint16,
    "u2": np.uint16,
    # numpy uint32
    np.uintc: np.uintc,
    "uintc": np.uintc,
    "I": np.uintc,
    "uint32": np.uint32,
    "u4": np.uint32,
    # numpy uint64
    np.uint: np.uint,
    "uint": np.uint,
    "uint64": np.uint64,
    "u8": np.uint64,
    # numpy extended uint
    np.ulonglong: np.ulonglong,
    "ulonglong": np.ulonglong,
    "Q": np.ulonglong,
    # numpy unsigned pointer  (platform dependent one of uint[8,16,32,64])
    np.uintp: np.uintp,
    "uintp": np.uintp,
    "P": np.uintp,
}
PYARROW_UINT_ARGS = {
    "uint8[pyarrow]": int,
    "uint16[pyarrow]": int,
    "uint32[pyarrow]": int,
    "uint64[pyarrow]": int,
}
ASTYPE_UINT_ARGS = PANDAS_UINT_ARGS | NUMPY_UINT_ARGS | PYARROW_UINT_ARGS

PYTHON_FLOAT_ARGS = {float: np.floating, "float": np.floating}
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
    "longdouble": np.longdouble,
    "f16": np.longdouble,  # NOTE: UNIX ONLY
    # "float96": np.longdouble,  # NOTE: unsupported
    "float128": np.longdouble,  # NOTE: UNIX ONLY
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
ASTYPE_FLOAT_NOT_NUMPY16_ARGS = (
    PYTHON_FLOAT_ARGS | NUMPY_FLOAT_NOT16_ARGS | PYARROW_FLOAT_ARGS | PANDAS_FLOAT_ARGS
)
ASTYPE_FLOAT_ARGS = ASTYPE_FLOAT_NOT_NUMPY16_ARGS | NUMPY_FLOAT16_ARGS

PYTHON_COMPLEX_ARGS = {complex: np.complexfloating, "complex": np.complexfloating}
NUMPY_COMPLEX_ARGS = {
    # numpy complex64
    np.csingle: np.csingle,
    "csingle": np.csingle,
    "F": np.csingle,
    "complex64": np.complex64,
    "c8": np.complex64,
    # numpy complex128
    np.cdouble: np.cdouble,
    "cdouble": np.cdouble,
    "D": np.cdouble,
    "complex128": np.complex128,
    "c16": np.complex128,
    # numpy complex256
    np.clongdouble: np.clongdouble,
    "clongdouble": np.clongdouble,
    "G": np.clongdouble,
    "c32": np.clongdouble,  # NOTE: UNIX ONLY
    # "complex192": np.clongdouble,  # NOTE: unsupported
    "complex256": np.clongdouble,  # NOTE: UNIX ONLY
}
ASTYPE_COMPLEX_ARGS = PYTHON_COMPLEX_ARGS | NUMPY_COMPLEX_ARGS

NUMPY_TIMESTAMP_ARGS = {
    # numpy datetime64
    "datetime64[Y]": datetime,
    "datetime64[M]": datetime,
    "datetime64[W]": datetime,
    "datetime64[D]": datetime,
    "datetime64[h]": datetime,
    "datetime64[m]": datetime,
    "datetime64[s]": datetime,
    "datetime64[ms]": datetime,
    "datetime64[us]": datetime,
    "datetime64[μs]": datetime,
    "datetime64[ns]": datetime,
    "datetime64[ps]": datetime,
    "datetime64[fs]": datetime,
    "datetime64[as]": datetime,
    # numpy datetime64 type codes
    "M8[Y]": datetime,
    "M8[M]": datetime,
    "M8[W]": datetime,
    "M8[D]": datetime,
    "M8[h]": datetime,
    "M8[m]": datetime,
    "M8[s]": datetime,
    "M8[ms]": datetime,
    "M8[us]": datetime,
    "M8[μs]": datetime,
    "M8[ns]": datetime,
    "M8[ps]": datetime,
    "M8[fs]": datetime,
    "M8[as]": datetime,
    # little endian
    "<M8[Y]": datetime,
    "<M8[M]": datetime,
    "<M8[W]": datetime,
    "<M8[D]": datetime,
    "<M8[h]": datetime,
    "<M8[m]": datetime,
    "<M8[s]": datetime,
    "<M8[ms]": datetime,
    "<M8[us]": datetime,
    "<M8[μs]": datetime,
    "<M8[ns]": datetime,
    "<M8[ps]": datetime,
    "<M8[fs]": datetime,
    "<M8[as]": datetime,
}
PYARROW_TIMESTAMP_ARGS = {
    # pyarrow timestamp
    "timestamp[s][pyarrow]": datetime,
    "timestamp[ms][pyarrow]": datetime,
    "timestamp[us][pyarrow]": datetime,
    "timestamp[ns][pyarrow]": datetime,
    # pyarrow date
    "date32[pyarrow]": date,
    "date64[pyarrow]": date,
}
ASTYPE_TIMESTAMP_ARGS = NUMPY_TIMESTAMP_ARGS | PYARROW_TIMESTAMP_ARGS

NUMPY_TIMEDELTA_ARGS = {
    # numpy timedelta64
    "timedelta64[Y]": timedelta,
    "timedelta64[M]": timedelta,
    "timedelta64[W]": timedelta,
    "timedelta64[D]": timedelta,
    "timedelta64[h]": timedelta,
    "timedelta64[m]": timedelta,
    "timedelta64[s]": timedelta,
    "timedelta64[ms]": timedelta,
    "timedelta64[us]": timedelta,
    "timedelta64[μs]": timedelta,
    "timedelta64[ns]": timedelta,
    "timedelta64[ps]": timedelta,
    "timedelta64[fs]": timedelta,
    "timedelta64[as]": timedelta,
    # numpy timedelta64 type codes
    "m8[Y]": timedelta,
    "m8[M]": timedelta,
    "m8[W]": timedelta,
    "m8[D]": timedelta,
    "m8[h]": timedelta,
    "m8[m]": timedelta,
    "m8[s]": timedelta,
    "m8[ms]": timedelta,
    "m8[us]": timedelta,
    "m8[μs]": timedelta,
    "m8[ns]": timedelta,
    "m8[ps]": timedelta,
    "m8[fs]": timedelta,
    "m8[as]": timedelta,
    # little endian
    "<m8[Y]": timedelta,
    "<m8[M]": timedelta,
    "<m8[W]": timedelta,
    "<m8[D]": timedelta,
    "<m8[h]": timedelta,
    "<m8[m]": timedelta,
    "<m8[s]": timedelta,
    "<m8[ms]": timedelta,
    "<m8[us]": timedelta,
    "<m8[μs]": timedelta,
    "<m8[ns]": timedelta,
    "<m8[ps]": timedelta,
    "<m8[fs]": timedelta,
    "<m8[as]": timedelta,
}
PYARROW_TIMEDELTA_ARGS = {
    # pyarrow duration
    "duration[s][pyarrow]": timedelta,
    "duration[ms][pyarrow]": timedelta,
    "duration[us][pyarrow]": timedelta,
    "duration[ns][pyarrow]": timedelta,
}
ASTYPE_TIMEDELTA_ARGS = NUMPY_TIMEDELTA_ARGS | PYARROW_TIMEDELTA_ARGS

PYTHON_STRING_ARGS = {str: str, "str": str}
PANDAS_STRING_ARGS = {pd.StringDtype(): str}
NUMPY_STRING_ARGS = {np.str_: str, "str_": str, "unicode": str, "U": str}
PYARROW_STRING_ARGS = {"string[pyarrow]": str}
ASTYPE_STRING_ARGS = (
    PYTHON_STRING_ARGS | PANDAS_STRING_ARGS | NUMPY_STRING_ARGS | PYARROW_STRING_ARGS
)

PYTHON_BYTES_ARGS = {bytes: bytes, "bytes": bytes}
NUMPY_BYTES_ARGS = {np.bytes_: np.bytes_, "bytes_": np.bytes_, "S": np.bytes_}
PYARROW_BYTES_ARGS = {"binary[pyarrow]": bytes}
ASTYPE_BYTES_ARGS = PYTHON_BYTES_ARGS | NUMPY_BYTES_ARGS | PYARROW_BYTES_ARGS

ASTYPE_CATEGORICAL_ARGS = {
    # pandas category
    pd.CategoricalDtype(): object,
    "category": object,
    # pyarrow dictionary
    # ("dictionary[pyarrow]", "pd.Series[category]", Categorical),
}

PYTHON_OBJECT_ARGS = {object: object, "object": object}
NUMPY_OBJECT_ARGS = {np.object_: object, "object_": object, "O": object}
ASTYPE_OBJECT_ARGS = PYTHON_OBJECT_ARGS | NUMPY_OBJECT_ARGS

NUMPY_VOID_ARGS = {np.void: np.void, "void": np.void, "V": np.void}
ASTYPE_VOID_ARGS = NUMPY_VOID_ARGS

PYTHON_NOT_DATETIME_DTYPE_ARGS = (
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
        value = actual.obj  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
    elif isinstance(actual, Iterable):
        value = next(iter(cast("Iterable[Any]", actual)))
    else:
        assert hasattr(actual, attr)
        value = getattr(actual, attr)

    if not isinstance(value, dtype):
        raise RuntimeError(
            f"Expected type '{dtype}' but got '{type(value)}'"  # pyright: ignore[reportUnknownArgumentType]
        )
    return actual


def pytest_warns_bounded(
    warning: type[Warning],
    match: str,
    lower: str | None = None,
    upper: str | None = None,
    version_str: str | None = None,
    upper_exception: type[Exception] | None = None,
) -> AbstractContextManager[Any]:
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
    if (WINDOWS or MAC) and dtype in {"f16", "float128", "c32", "complex256"}:
        return TypeError
    return None
