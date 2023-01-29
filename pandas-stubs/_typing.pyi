from builtins import type as type_t
from collections.abc import (
    Callable,
    Hashable,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
import datetime
from os import PathLike
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

import numpy as np
from numpy import typing as npt
from pandas.core.arrays import ExtensionArray
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from typing_extensions import TypeAlias

from pandas._libs.interval import Interval
from pandas._libs.tslibs import (
    Period,
    Timedelta,
    Timestamp,
)

from pandas.core.dtypes.dtypes import ExtensionDtype

from pandas.io.formats.format import EngFormatter

ArrayLike: TypeAlias = Union[ExtensionArray, np.ndarray]
AnyArrayLike: TypeAlias = Union[Index, Series, np.ndarray]
PythonScalar: TypeAlias = Union[str, bool, complex]
DatetimeLikeScalar = TypeVar("DatetimeLikeScalar", Period, Timestamp, Timedelta)
PandasScalar: TypeAlias = Union[
    bytes, datetime.date, datetime.datetime, datetime.timedelta
]
# Scalar: TypeAlias = Union[PythonScalar, PandasScalar]

DatetimeLike: TypeAlias = Union[datetime.datetime, np.datetime64, Timestamp]
DateAndDatetimeLike: TypeAlias = Union[datetime.date, DatetimeLike]

DatetimeDictArg: TypeAlias = Union[
    Sequence[int], Sequence[float], list[str], tuple[Scalar, ...], AnyArrayLike
]
DictConvertible: TypeAlias = Union[FulldatetimeDict, DataFrame]

class YearMonthDayDict(TypedDict, total=True):
    year: DatetimeDictArg
    month: DatetimeDictArg
    day: DatetimeDictArg

class FulldatetimeDict(YearMonthDayDict, total=False):
    hour: DatetimeDictArg
    hours: DatetimeDictArg
    minute: DatetimeDictArg
    minutes: DatetimeDictArg
    second: DatetimeDictArg
    seconds: DatetimeDictArg
    ms: DatetimeDictArg
    us: DatetimeDictArg
    ns: DatetimeDictArg

# dtypes
NpDtype: TypeAlias = Union[
    str, np.dtype[np.generic], type[Union[str, complex, bool, object]]
]
Dtype: TypeAlias = Union[ExtensionDtype, NpDtype]
AstypeArg: TypeAlias = Union[ExtensionDtype, npt.DTypeLike]
# DtypeArg specifies all allowable dtypes in a functions its dtype argument
DtypeArg: TypeAlias = Union[Dtype, dict[Any, Dtype]]
DtypeObj: TypeAlias = Union[np.dtype[np.generic], ExtensionDtype]

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

FilePath: TypeAlias = Union[str, PathLike[str]]

Axis: TypeAlias = Union[str, int]
IndexLabel: TypeAlias = Union[Hashable, Sequence[Hashable]]
Label: TypeAlias = Optional[Hashable]
Level: TypeAlias = Union[Hashable, int]
Suffixes: TypeAlias = tuple[Optional[str], Optional[str]]
Ordered: TypeAlias = Optional[bool]
JSONSerializable: TypeAlias = Union[PythonScalar, list, dict]
Axes: TypeAlias = Union[AnyArrayLike, list, dict, range, tuple]
Renamer: TypeAlias = Union[Mapping[Any, Label], Callable[[Any], Label]]
T = TypeVar("T")
FuncType: TypeAlias = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
HashableT = TypeVar("HashableT", bound=Hashable)
HashableT1 = TypeVar("HashableT1", bound=Hashable)
HashableT2 = TypeVar("HashableT2", bound=Hashable)
HashableT3 = TypeVar("HashableT3", bound=Hashable)
HashableT4 = TypeVar("HashableT4", bound=Hashable)
HashableT5 = TypeVar("HashableT5", bound=Hashable)

AggFuncTypeBase: TypeAlias = Union[Callable, str, np.ufunc]
AggFuncTypeDictSeries: TypeAlias = Mapping[HashableT, AggFuncTypeBase]
AggFuncTypeDictFrame: TypeAlias = Mapping[
    HashableT, Union[AggFuncTypeBase, list[AggFuncTypeBase]]
]
AggFuncTypeSeriesToFrame: TypeAlias = Union[
    list[AggFuncTypeBase],
    AggFuncTypeDictSeries,
]
AggFuncTypeFrame: TypeAlias = Union[
    AggFuncTypeBase,
    list[AggFuncTypeBase],
    AggFuncTypeDictFrame,
]

num: TypeAlias = complex
SeriesAxisType: TypeAlias = Literal[
    "index", 0
]  # Restricted subset of _AxisType for series
AxisTypeIndex: TypeAlias = Literal["index", 0]
AxisTypeColumn: TypeAlias = Literal["columns", 1]
AxisType: TypeAlias = AxisTypeIndex | AxisTypeColumn
DtypeNp = TypeVar("DtypeNp", bound=np.dtype[np.generic])
KeysArgType: TypeAlias = Any
ListLike = TypeVar("ListLike", Sequence, np.ndarray, "Series", "Index")
ListLikeExceptSeriesAndStr = TypeVar(
    "ListLikeExceptSeriesAndStr", MutableSequence, np.ndarray, tuple, "Index"
)
ListLikeU: TypeAlias = Union[Sequence, np.ndarray, Series, Index]
StrLike: TypeAlias = Union[str, np.str_]
IndexIterScalar: TypeAlias = Union[
    str,
    bytes,
    datetime.date,
    datetime.datetime,
    datetime.timedelta,
    bool,
    int,
    float,
    Timestamp,
    Timedelta,
]
Scalar: TypeAlias = Union[
    IndexIterScalar,
    complex,
]
ScalarT = TypeVar("ScalarT", bound=Scalar)
# Refine the definitions below in 3.9 to use the specialized type.
np_ndarray_int64: TypeAlias = npt.NDArray[np.int64]
np_ndarray_int: TypeAlias = npt.NDArray[np.signedinteger]
np_ndarray_anyint: TypeAlias = npt.NDArray[np.integer]
np_ndarray_bool: TypeAlias = npt.NDArray[np.bool_]
np_ndarray_str: TypeAlias = npt.NDArray[np.str_]

IndexType: TypeAlias = Union[slice, np_ndarray_anyint, Index, list[int], Series[int]]
MaskType: TypeAlias = Union[Series[bool], np_ndarray_bool, list[bool]]
# Scratch types for generics
S1 = TypeVar(
    "S1",
    str,
    bytes,
    datetime.date,
    datetime.time,
    bool,
    int,
    float,
    complex,
    Timestamp,
    Timedelta,
    np.datetime64,
    Period,
    Interval[int],
    Interval[float],
    Interval[Timestamp],
    Interval[Timedelta],
)
T1 = TypeVar(
    "T1", str, int, np.int64, np.uint64, np.float64, float, np.dtype[np.generic]
)
T2 = TypeVar("T2", str, int)

IndexingInt: TypeAlias = Union[
    int, np.int_, np.integer, np.unsignedinteger, np.signedinteger, np.int8
]
TimestampConvertibleTypes: TypeAlias = Union[
    Timestamp, datetime.datetime, datetime.date, np.datetime64, np.int64, float, str
]
TimedeltaConvertibleTypes: TypeAlias = Union[
    Timedelta, datetime.timedelta, np.timedelta64, np.int64, float, str
]
# NDFrameT is stricter and ensures that the same subclass of NDFrame always is
# used. E.g. `def func(a: NDFrameT) -> NDFrameT: ...` means that if a
# Series is passed into a function, a Series is always returned and if a DataFrame is
# passed in, a DataFrame is always returned.
NDFrameT = TypeVar("NDFrameT", bound=NDFrame)

IndexT = TypeVar("IndexT", bound=Index)

# Interval closed type
IntervalT = TypeVar(
    "IntervalT",
    Interval[int],
    Interval[float],
    Interval[Timestamp],
    Interval[Timedelta],
)
IntervalClosedType: TypeAlias = Literal["left", "right", "both", "neither"]

IgnoreRaiseCoerce: TypeAlias = Literal["ignore", "raise", "coerce"]

# Shared by functions such as drop and astype
IgnoreRaise: TypeAlias = Literal["ignore", "raise"]

# for arbitrary kwargs passed during reading/writing files
StorageOptions: TypeAlias = Optional[dict[str, Any]]

# compression keywords and compression
CompressionDict: TypeAlias = dict[str, Any]
CompressionOptions: TypeAlias = Optional[
    Union[Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"], CompressionDict]
]
FormattersType: TypeAlias = Union[
    list[Callable], tuple[Callable, ...], Mapping[Union[str, int], Callable]
]
FloatFormatType: TypeAlias = str | Callable | EngFormatter
# converters
ConvertersArg: TypeAlias = dict[Hashable, Callable[[Dtype], Dtype]]

# parse_dates
ParseDatesArg: TypeAlias = Union[
    bool, list[Hashable], list[list[Hashable]], dict[Hashable, list[Hashable]]
]

# read_xml parsers
XMLParsers: TypeAlias = Literal["lxml", "etree"]

# Any plain Python or numpy function
Function: TypeAlias = Union[np.ufunc, Callable[..., Any]]
# Use a distinct HashableT in shared types to avoid conflicts with
# shared HashableT and HashableT#. This one can be used if the identical
# type is need in a function that uses GroupByObjectNonScalar
_HashableTa = TypeVar("_HashableTa", bound=Hashable)
GroupByObjectNonScalar: TypeAlias = Union[
    tuple,
    list[_HashableTa],
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
GroupByObject: TypeAlias = Union[Scalar, GroupByObjectNonScalar]

StataDateFormat: TypeAlias = Literal[
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

FillnaOptions: TypeAlias = Literal["backfill", "bfill", "ffill", "pad"]
ReplaceMethod: TypeAlias = Literal["pad", "ffill", "bfill"]
SortKind: TypeAlias = Literal["quicksort", "mergesort", "heapsort", "stable"]
NaPosition: TypeAlias = Literal["first", "last"]
JoinHow: TypeAlias = Literal["left", "right", "outer", "inner"]
MergeHow: TypeAlias = Union[JoinHow, Literal["cross"]]
JsonFrameOrient: TypeAlias = Literal[
    "split", "records", "index", "columns", "values", "table"
]
JsonSeriesOrient: TypeAlias = Literal["split", "records", "index", "table"]

TimestampConvention: TypeAlias = Literal["start", "end", "s", "e"]

CSVEngine: TypeAlias = Literal["c", "python", "pyarrow", "python-fwf"]
CSVQuoting: TypeAlias = Literal[0, 1, 2, 3]

HDFCompLib: TypeAlias = Literal["zlib", "lzo", "bzip2", "blosc"]
ParquetEngine: TypeAlias = Literal["auto", "pyarrow", "fastparquet"]
FileWriteMode: TypeAlias = Literal[
    "a", "w", "x", "at", "wt", "xt", "ab", "wb", "xb", "w+", "w+b", "a+", "a+b"
]
ColspaceArgType: TypeAlias = (
    str | int | Sequence[int | str] | Mapping[Hashable, str | int]
)

# Windowing rank methods
WindowingRankType: TypeAlias = Literal["average", "min", "max"]
WindowingEngine: TypeAlias = Union[Literal["cython", "numba"], None]

class _WindowingNumbaKwargs(TypedDict, total=False):
    nopython: bool
    nogil: bool
    parallel: bool

WindowingEngineKwargs: TypeAlias = Union[_WindowingNumbaKwargs, None]
QuantileInterpolation: TypeAlias = Literal[
    "linear", "lower", "higher", "midpoint", "nearest"
]

class StyleExportDict(TypedDict, total=False):
    apply: Any
    table_attributes: Any
    table_styles: Any
    hide_index: bool
    hide_columns: bool
    hide_index_names: bool
    hide_column_names: bool
    css: dict[str, str | int]

CalculationMethod: TypeAlias = Literal["single", "table"]

ValidationOptions: TypeAlias = Literal[
    "one_to_one",
    "1:1",
    "one_to_many",
    "1:m",
    "many_to_one",
    "m:1",
    "many_to_many",
    "m:m",
]
__all__ = ["npt", "type_t"]
