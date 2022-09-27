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
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
from numpy import typing as npt
from pandas.core.arrays import ExtensionArray
from pandas.core.generic import NDFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series

from pandas._libs.tslibs import (
    Period,
    Timedelta,
    Timestamp,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

from pandas.io.formats.format import EngFormatter

ArrayLike = Union[ExtensionArray, np.ndarray]
AnyArrayLike = Union[Index, Series, np.ndarray]
PythonScalar = Union[str, bool, complex]
DatetimeLikeScalar = TypeVar("DatetimeLikeScalar", Period, Timestamp, Timedelta)
PandasScalar = Union[bytes, datetime.date, datetime.datetime, datetime.timedelta]
# Scalar = Union[PythonScalar, PandasScalar]

DatetimeLike = Union[datetime.date, datetime.datetime, np.datetime64, Timestamp]

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

class BaseBuffer(Protocol):
    @property
    def mode(self) -> str: ...
    def seek(self, __offset: int, __whence: int = ...) -> int: ...
    def seekable(self) -> bool: ...
    def tell(self) -> int: ...

class ReadBuffer(BaseBuffer, Protocol[AnyStr_cov]):
    def read(self, __n: int = ...) -> AnyStr_cov: ...

class WriteBuffer(BaseBuffer, Protocol[AnyStr_con]):
    def write(self, __b: AnyStr_con) -> Any: ...
    def flush(self) -> Any: ...

class ReadPickleBuffer(ReadBuffer[bytes], Protocol):
    def readline(self) -> bytes: ...

class ReadCsvBuffer(ReadBuffer[AnyStr_cov], Protocol[AnyStr_cov]):
    def __iter__(self) -> Iterator[AnyStr_cov]: ...
    def fileno(self) -> int: ...
    def readline(self) -> AnyStr_cov: ...
    @property
    def closed(self) -> bool: ...

class WriteExcelBuffer(WriteBuffer[bytes], Protocol):
    def truncate(self, size: int | None = ...) -> int: ...

FilePath = Union[str, PathLike[str]]
FilePathOrBuffer = Union[
    FilePath, IO[AnyStr], RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper, mmap
]

Axis = Union[str, int]
IndexLabel = Union[Hashable, Sequence[Hashable]]
Label = Optional[Hashable]
Level = Union[Hashable, int]
Suffixes = tuple[Optional[str], Optional[str]]
Ordered = Optional[bool]
JSONSerializable = Union[PythonScalar, list, dict]
Axes = Union[AnyArrayLike, list, dict, range]
Renamer = Union[Mapping[Any, Label], Callable[[Any], Label]]
T = TypeVar("T")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
HashableT = TypeVar("HashableT", bound=Hashable)

AggFuncTypeBase = Union[Callable, str, np.ufunc]
AggFuncTypeDictSeries = dict[Hashable, AggFuncTypeBase]
AggFuncTypeDictFrame = dict[Hashable, Union[AggFuncTypeBase, list[AggFuncTypeBase]]]
AggFuncTypeSeriesToFrame = Union[
    list[AggFuncTypeBase],
    AggFuncTypeDictSeries,
]
AggFuncTypeFrame = Union[
    AggFuncTypeBase,
    list[AggFuncTypeBase],
    AggFuncTypeDictFrame,
]

num = complex
SeriesAxisType = Literal["index", 0]  # Restricted subset of _AxisType for series
AxisType = Literal["columns", "index", 0, 1]
DtypeNp = TypeVar("DtypeNp", bound=np.dtype[np.generic])
KeysArgType = Any
ListLike = TypeVar("ListLike", Sequence, np.ndarray, "Series", "Index")
ListLikeU = Union[Sequence, np.ndarray, Series, Index]
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
np_ndarray_int64 = npt.NDArray[np.int64]
np_ndarray_int = npt.NDArray[np.signedinteger]
np_ndarray_anyint = npt.NDArray[np.integer]
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
TimestampConvertibleTypes = Union[
    Timestamp, datetime.datetime, np.datetime64, np.int64, float, str
]
TimedeltaConvertibleTypes = Union[
    Timedelta, datetime.timedelta, np.timedelta64, np.int64, float, str
]
# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound=NDFrame)

IndexT = TypeVar("IndexT", bound=Index)

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
FormattersType = Union[
    list[Callable], tuple[Callable, ...], Mapping[Union[str, int], Callable]
]
FloatFormatType = str | Callable | EngFormatter
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
    tuple,
    list[HashableT],
    Function,
    list[Function],
    Series,
    list[Series],
    np.ndarray,
    list[np.ndarray],
    Mapping[Label, Any],
    list[Mapping[Label, Any]],
    Index,
    list[Index],
    Grouper,
    list[Grouper],
]
GroupByObject = Union[Scalar, GroupByObjectNonScalar]

StataDateFormat = Literal[
    "tc",
    "%tc",
    "td",
    "%td",
    "tw",
    "%tw",
    "tm",
    "%tm",
    "tq",
    "%tq",
    "th",
    "%th",
    "ty",
    "%ty",
]

FillnaOptions = Literal["backfill", "bfill", "ffill", "pad"]
ReplaceMethod = Literal["pad", "ffill", "bfill"]
SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition = Literal["first", "last"]
MergeHow = Literal["left", "right", "outer", "inner"]
JsonFrameOrient = Literal["split", "records", "index", "columns", "values", "table"]
JsonSeriesOrient = Literal["split", "records", "index"]

TimestampConvention = Literal["start", "end", "s", "e"]

CSVEngine = Literal["c", "python", "pyarrow", "python-fwf"]
CSVQuoting = Literal[0, 1, 2, 3]

HDFCompLib = Literal["zlib", "lzo", "bzip2", "blosc"]
ParquetEngine = Literal["auto", "pyarrow", "fastparquet"]
FileWriteMode = Literal[
    "a", "w", "x", "at", "wt", "xt", "ab", "wb", "xb", "w+", "w+b", "a+", "a+b"
]
ColspaceArgType = str | int | Sequence[int | str] | Mapping[Hashable, str | int]

# Windowing rank methods
WindowingRankType = Literal["average", "min", "max"]
WindowingEngine = Union[Literal["cython", "numba"], None]

class _WindowingNumbaKwargs(TypedDict, total=False):
    nopython: bool
    nogil: bool
    parallel: bool

WindowingEngineKwargs = Union[_WindowingNumbaKwargs, None]
QuantileInterpolation = Literal["linear", "lower", "higher", "midpoint", "nearest"]

class StyleExportDict(TypedDict, total=False):
    apply: Any
    table_attributes: Any
    table_styles: Any
    hide_index: bool
    hide_columns: bool
    hide_index_names: bool
    hide_column_names: bool
    css: dict[str, str | int]

CalculationMethod = Literal["single", "table"]

__all__ = ["npt", "type_t"]
