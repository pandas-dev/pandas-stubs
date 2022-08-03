from builtins import type as type_t
import datetime
from io import (
    BufferedIOBase,
    RawIOBase,
    TextIOBase,
    TextIOWrapper,
)
from mmap import mmap
from os import PathLike
from typing import (
    IO,
    Any,
    AnyStr,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
from numpy import typing as npt
from pandas.core.arrays import ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.indexes.base import Index
from pandas.core.series import Series

from pandas._libs.tslibs import (
    Period,
    Timedelta,
    Timestamp,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

ArrayLike = Union[ExtensionArray, np.ndarray]
AnyArrayLike = Union[Index, Series, np.ndarray]
PythonScalar = Union[str, bool, complex]
DatetimeLikeScalar = TypeVar("DatetimeLikeScalar", Period, Timestamp, Timedelta)
PandasScalar = Union[bytes, datetime.date, datetime.datetime, datetime.timedelta]
# Scalar = Union[PythonScalar, PandasScalar]
IntStrT = TypeVar("IntStrT", int, str)

# dtypes
NpDtype = Union[str, np.dtype[np.generic], type[Union[str, complex, bool, object]]]
Dtype = Union[ExtensionDtype, NpDtype]
AstypeArg = Union[ExtensionDtype, npt.DTypeLike]
# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg = Union[Dtype, dict[Any, Dtype]]
DtypeObj = Union[np.dtype[np.generic], ExtensionDtype]

# filenames and file-like-objects
AnyStr_cov = TypeVar("AnyStr_cov", str, bytes, covariant=True)
AnyStr_con = TypeVar("AnyStr_con", str, bytes, contravariant=True)

class BaseBuffer(Protocol): ...
class ReadBuffer(BaseBuffer, Protocol[AnyStr_cov]): ...
class WriteBuffer(BaseBuffer, Protocol[AnyStr_cov]): ...

FilePath = Union[str, PathLike[str]]

Buffer = Union[IO[AnyStr], RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap]
FileOrBuffer = Union[str, Buffer[AnyStr]]
FilePathOrBuffer = Union[PathLike[str], FileOrBuffer[AnyStr]]
FilePathOrBytesBuffer = Union[PathLike[str], WriteBuffer[bytes]]

FrameOrSeries = TypeVar("FrameOrSeries", bound=NDFrame)
FrameOrSeriesUnion = Union[DataFrame, Series]
Axis = Union[str, int]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Label = Optional[Hashable]
Level = Union[Hashable, int]
Ordered = Optional[bool]
JSONSerializable = Union[PythonScalar, list, dict]
Axes = Union[AnyArrayLike, list, dict, range]
Renamer = Union[Mapping[Any, Label], Callable[[Any], Label]]
T = TypeVar("T")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
HashableT = TypeVar("HashableT", bound=Hashable)

AggFuncTypeBase = Union[Callable, str]
AggFuncTypeDict = dict[Hashable, Union[AggFuncTypeBase, list[AggFuncTypeBase]]]
AggFuncType = Union[
    AggFuncTypeBase,
    list[AggFuncTypeBase],
    AggFuncTypeDict,
]

num = complex
SeriesAxisType = Literal["index", 0]  # Restricted subset of _AxisType for series
AxisType = Literal["columns", "index", 0, 1]
DtypeNp = TypeVar("DtypeNp", bound=np.dtype[np.generic])
KeysArgType = Any
ListLike = TypeVar("ListLike", Sequence, np.ndarray, "Series")
StrLike = Union[str, np.str_]
Scalar = Union[
    str,
    bytes,
    datetime.date,
    datetime.datetime,
    datetime.timedelta,
    bool,
    complex,
    Timestamp,
    Timedelta,
]
ScalarT = TypeVar("ScalarT", bound=Scalar)
# Refine the definitions below in 3.9 to use the specialized type.
np_ndarray_int8 = npt.NDArray[np.int8]
np_ndarray_int16 = npt.NDArray[np.int16]
np_ndarray_int32 = npt.NDArray[np.int32]
np_ndarray_int64 = npt.NDArray[np.int64]
np_ndarray_uint8 = npt.NDArray[np.uint8]
np_ndarray_uint16 = npt.NDArray[np.uint16]
np_ndarray_uint32 = npt.NDArray[np.uint32]
np_ndarray_uint64 = npt.NDArray[np.uint64]
np_ndarray_int = Union[
    np_ndarray_int8, np_ndarray_int16, np_ndarray_int32, np_ndarray_int64
]
np_ndarray_uint = Union[
    np_ndarray_uint8, np_ndarray_uint16, np_ndarray_uint32, np_ndarray_uint64
]
np_ndarray_anyint = Union[np_ndarray_int, np_ndarray_uint]
np_ndarray_bool = npt.NDArray[np.bool_]
np_ndarray_str = npt.NDArray[np.str_]

IndexType = Union[slice, np_ndarray_int64, Index, list[int], Series[int]]
MaskType = Union[Series[bool], np_ndarray_bool, list[bool]]
# Scratch types for generics
S1 = TypeVar(
    "S1",
    str,
    bytes,
    datetime.date,
    datetime.datetime,
    datetime.time,
    datetime.timedelta,
    bool,
    int,
    float,
    complex,
    Timestamp,
    Timedelta,
    np.datetime64,
    Period,
)
T1 = TypeVar(
    "T1", str, int, np.int64, np.uint64, np.float64, float, np.dtype[np.generic]
)
T2 = TypeVar("T2", str, int)

IndexingInt = Union[
    int, np.int_, np.integer, np.unsignedinteger, np.signedinteger, np.int8
]

# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound=NDFrame)

# Interval closed type

IntervalClosedType = Literal["left", "right", "both", "neither"]

DateTimeErrorChoices = Literal["ignore", "raise", "coerce"]

# Shared by functions such as drop and astype
IgnoreRaise = Literal["ignore", "raise"]

# for arbitrary kwargs passed during reading/writing files
StorageOptions = Optional[dict[str, Any]]

# compression keywords and compression
CompressionDict = dict[str, Any]
CompressionOptions = Optional[
    Union[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"], CompressionDict]
]

# converters
ConvertersArg = dict[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg = Union[
    bool, list[Hashable], list[list[Hashable]], dict[Hashable, list[Hashable]]
]

# read_xml parsers
XMLParsers = Literal["lxml", "etree"]

# Any plain Python or numpy function
Function = Union[np.ufunc, Callable[..., Any]]
GroupByObjectNonScalar = Union[
    tuple, list[Label], Function, Series, np.ndarray, Mapping[Label, Any], Index
]
GroupByObject = Union[Scalar, GroupByObjectNonScalar]

__all__ = ["npt", "type_t"]
