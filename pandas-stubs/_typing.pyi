import datetime
from io import (
    BufferedIOBase,
    RawIOBase,
    TextIOBase,
    TextIOWrapper,
)
from mmap import mmap
from os import PathLike
from pathlib import Path
import sys
from typing import (
    IO,
    Any,
    AnyStr,
    Callable,
    Collection,
    Dict,
    Hashable,
    List,
    Literal,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from numpy import typing as npt
from pandas.core.arrays import ExtensionArray as ExtensionArray
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.generic import NDFrame
from pandas.core.indexes.base import Index as Index
from pandas.core.series import Series as Series

from pandas._libs.tslibs import (
    Period,
    Timedelta as Timedelta,
    Timestamp as Timestamp,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

ArrayLike = Union[ExtensionArray, np.ndarray]
AnyArrayLike = Union[Index, Series, np.ndarray]
PythonScalar = Union[str, int, float, bool, complex]
DatetimeLikeScalar = TypeVar("DatetimeLikeScalar", Period, Timestamp, Timedelta)
PandasScalar = Union[bytes, datetime.date, datetime.datetime, datetime.timedelta]
# Scalar = Union[PythonScalar, PandasScalar]
IntStrT = TypeVar("IntStrT", int, str)

# dtypes
NpDtype = Union[
    str, np.dtype[np.generic], Type[Union[str, float, int, complex, bool, object]]
]
Dtype = Union[ExtensionDtype, NpDtype]
AstypeArg = Union[ExtensionDtype, npt.DTypeLike]
# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg = Union[Dtype, Dict[Any, Dtype]]
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
JSONSerializable = Union[PythonScalar, List, Dict]
Axes = Union[AnyArrayLike, List, Dict, range]
Renamer = Union[Mapping[Any, Label], Callable[[Any], Label]]
T = TypeVar("T")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)

AggFuncTypeBase = Union[Callable, str]
AggFuncTypeDict = Dict[Hashable, Union[AggFuncTypeBase, List[AggFuncTypeBase]]]
AggFuncType = Union[
    AggFuncTypeBase,
    List[AggFuncTypeBase],
    AggFuncTypeDict,
]

num = Union[int, float]
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
    int,
    float,
    complex,
    Timestamp,
    Timedelta,
]
# Refine the next 3 in 3.9 to use the specialized type.
np_ndarray_int64 = npt.NDArray[np.int64]
np_ndarray_bool = npt.NDArray[np.bool_]
np_ndarray_str = npt.NDArray[np.str_]
IndexType = Union[slice, np_ndarray_int64, Index, List[int], Series[int]]
MaskType = Union[Series[bool], np_ndarray_bool, List[bool]]
# Scratch types for generics
S1 = TypeVar(
    "S1",
    str,
    bytes,
    datetime.date,
    datetime.datetime,
    datetime.timedelta,
    bool,
    int,
    float,
    complex,
    Timestamp,
    Timedelta,
    np.datetime64,
)
T1 = TypeVar(
    "T1", str, int, np.int64, np.uint64, np.float64, float, np.dtype[np.generic]
)
T2 = TypeVar("T2", str, int)

IndexingInt = Union[
    int, np.int_, np.integer, np.unsignedinteger, np.signedinteger, np.int8
]

# Interval closed type

IntervalClosedType = Literal["left", "right", "both", "neither"]

DateTimeErrorChoices = Literal["ignore", "raise", "coerce"]

# Shared by functions such as drop and astype
IgnoreRaise = Literal["ignore", "raise"]

# for arbitrary kwargs passed during reading/writing files
StorageOptions = Optional[Dict[str, Any]]

# compression keywords and compression
CompressionDict = Dict[str, Any]
CompressionOptions = Optional[
    Union[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"], CompressionDict]
]

# converters
ConvertersArg = Dict[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg = Union[
    bool, List[Hashable], List[List[Hashable]], Dict[Hashable, List[Hashable]]
]

# read_xml parsers
XMLParsers = Literal["lxml", "etree"]

# Any plain Python or numpy function
Function = Union[np.ufunc, Callable[..., Any]]
GroupByObject = Union[
    Label, List[Label], Function, Series, np.ndarray, Mapping[Label, Any]
]
