from builtins import (
    bool as _bool,
    str as _str,
)
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    Sequence,
    ValuesView,
)
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NoReturn,
    TypeVar,
    final,
    overload,
    type_check_only,
)

from matplotlib.axes import (
    Axes as PlotAxes,
    SubplotBase,
)
import numpy as np
from pandas import (
    Index,
    Period,
    PeriodDtype,
    Timedelta,
    Timestamp,
)
from pandas.core.api import (
    Int8Dtype as Int8Dtype,
    Int16Dtype as Int16Dtype,
    Int32Dtype as Int32Dtype,
    Int64Dtype as Int64Dtype,
)
from pandas.core.arrays import TimedeltaArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.arrays.datetimes import DatetimeArray
from pandas.core.arrays.interval import IntervalArray
from pandas.core.base import IndexOpsMixin
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.groupby.groupby import BaseGroupBy
from pandas.core.indexers import BaseIndexer
from pandas.core.indexes.accessors import (
    CombinedDatetimelikeProperties,
    PeriodProperties,
    TimedeltaProperties,
    TimestampProperties,
)
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.interval import IntervalIndex
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexing import (
    _AtIndexer,
    _iAtIndexer,
    _iLocIndexer,
    _IndexSliceTuple,
    _LocIndexer,
)
from pandas.core.strings.accessor import StringMethods
from pandas.core.window import (
    Expanding,
    ExponentialMovingWindow,
)
from pandas.core.window.rolling import (
    Rolling,
    Window,
)
from typing_extensions import (
    Never,
    Self,
    TypeAlias,
)
import xarray as xr

from pandas._libs.interval import (
    Interval,
    _OrderableT,
)
from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._libs.missing import NAType
from pandas._libs.tslibs import BaseOffset
from pandas._libs.tslibs.nattype import NaTType
from pandas._libs.tslibs.offsets import DateOffset
from pandas._typing import (
    S1,
    S2,
    AggFuncTypeBase,
    AggFuncTypeDictFrame,
    AggFuncTypeSeriesToFrame,
    AnyAll,
    AnyArrayLike,
    ArrayLike,
    Axes,
    AxesData,
    Axis,
    AxisColumn,
    AxisIndex,
    BooleanDtypeArg,
    BytesDtypeArg,
    CalculationMethod,
    CategoryDtypeArg,
    ComplexDtypeArg,
    CompressionOptions,
    DropKeep,
    Dtype,
    DTypeLike,
    DtypeObj,
    FilePath,
    FillnaOptions,
    FloatDtypeArg,
    FloatFormatType,
    GenericT,
    GenericT_co,
    GroupByObjectNonScalar,
    HashableT1,
    IgnoreRaise,
    IndexingInt,
    IndexKeyFunc,
    IndexLabel,
    IntDtypeArg,
    InterpolateOptions,
    IntervalClosedType,
    IntervalT,
    JoinHow,
    JSONSerializable,
    JsonSeriesOrient,
    Just,
    Label,
    Level,
    ListLike,
    ListLikeU,
    MaskType,
    NaPosition,
    NsmallestNlargestKeep,
    ObjectDtypeArg,
    QuantileInterpolation,
    RandomState,
    ReindexMethod,
    Renamer,
    ReplaceValue,
    Scalar,
    ScalarT,
    SequenceNotStr,
    SeriesByT,
    SortKind,
    StrDtypeArg,
    StrLike,
    Suffixes,
    SupportsDType,
    T as _T,
    TimeAmbiguous,
    TimedeltaDtypeArg,
    TimestampDtypeArg,
    TimeUnit,
    TimeZones,
    ToTimestampHow,
    UIntDtypeArg,
    ValueKeyFunc,
    VoidDtypeArg,
    WriteBuffer,
    np_1darray,
    np_ndarray_anyint,
    np_ndarray_bool,
    np_ndarray_complex,
    np_ndarray_dt,
    np_ndarray_float,
    np_ndarray_str,
    np_ndarray_td,
    npt,
    num,
)

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.plotting import PlotAccessor

_T_INT = TypeVar("_T_INT", bound=int)
_T_COMPLEX = TypeVar("_T_COMPLEX", bound=complex)

class _iLocIndexerSeries(_iLocIndexer, Generic[S1]):
    # get item
    @overload
    def __getitem__(self, idx: IndexingInt) -> S1: ...
    @overload
    def __getitem__(self, idx: Index | slice | np_ndarray_anyint) -> Series[S1]: ...
    # set item
    @overload
    def __setitem__(self, idx: int, value: S1 | None) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: Index | slice | np_ndarray_anyint | list[int],
        value: S1 | Series[S1] | None,
    ) -> None: ...

class _LocIndexerSeries(_LocIndexer, Generic[S1]):
    # ignore needed because of mypy.  Overlapping, but we want to distinguish
    # having a tuple of just scalars, versus tuples that include slices or Index
    @overload
    def __getitem__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        self,
        idx: Scalar | tuple[Scalar, ...],
        # tuple case is for getting a specific element when using a MultiIndex
    ) -> S1: ...
    @overload
    def __getitem__(
        self,
        idx: (
            MaskType
            | Index
            | SequenceNotStr[float | _str | Timestamp]
            | slice
            | _IndexSliceTuple
            | Sequence[_IndexSliceTuple]
            | Callable
        ),
        # _IndexSliceTuple is when having a tuple that includes a slice.  Could just
        # be s.loc[1, :], or s.loc[pd.IndexSlice[1, :]]
    ) -> Series[S1]: ...
    @overload
    def __setitem__(
        self,
        idx: Index | MaskType | slice,
        value: S1 | ArrayLike | Series[S1] | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: _str,
        value: S1 | None,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: MaskType | StrLike | _IndexSliceTuple | list[ScalarT],
        value: S1 | ArrayLike | Series[S1] | None,
    ) -> None: ...

_ListLike: TypeAlias = (
    ArrayLike | dict[_str, np.ndarray] | Sequence[S1] | IndexOpsMixin[S1]
)

class Series(IndexOpsMixin[S1], NDFrame):
    # Define __index__ because mypy thinks Series follows protocol `SupportsIndex` https://github.com/pandas-dev/pandas-stubs/pull/1332#discussion_r2285648790
    __index__: ClassVar[None]
    __hash__: ClassVar[None]

    @overload
    def __new__(
        cls,
        data: npt.NDArray[np.float64],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[float]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[Never],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series: ...
    @overload
    def __new__(
        cls,
        data: Sequence[list[_str]],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[list[_str]]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[_str],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[_str]: ...
    @overload
    def __new__(
        cls,
        data: (
            DatetimeIndex
            | Sequence[np.datetime64 | datetime | date]
            | dict[HashableT1, np.datetime64 | datetime | date]
            | np.datetime64
            | datetime
            | date
        ),
        index: AxesData | None = ...,
        dtype: TimestampDtypeArg = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> TimestampSeries: ...
    @overload
    def __new__(
        cls,
        data: _ListLike,
        index: AxesData | None = ...,
        *,
        dtype: TimestampDtypeArg,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> TimestampSeries: ...
    @overload
    def __new__(
        cls,
        data: PeriodIndex | Sequence[Period],
        index: AxesData | None = ...,
        dtype: PeriodDtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> PeriodSeries: ...
    @overload
    def __new__(
        cls,
        data: (
            TimedeltaIndex
            | Sequence[np.timedelta64 | timedelta]
            | dict[HashableT1, np.timedelta64 | timedelta]
            | np.timedelta64
            | timedelta
        ),
        index: AxesData | None = ...,
        dtype: TimedeltaDtypeArg = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> TimedeltaSeries: ...
    @overload
    def __new__(
        cls,
        data: (
            IntervalIndex[Interval[_OrderableT]]
            | Interval[_OrderableT]
            | Sequence[Interval[_OrderableT]]
            | dict[HashableT1, Interval[_OrderableT]]
        ),
        index: AxesData | None = ...,
        dtype: Literal["Interval"] = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> IntervalSeries[_OrderableT]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        data: Scalar | _ListLike | dict[HashableT1, Any] | None,
        index: AxesData | None = ...,
        *,
        dtype: type[S1],
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(  # type: ignore[overload-overlap] # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[bool],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[bool]: ...
    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        data: Sequence[int],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[int]: ...
    @overload
    def __new__(
        cls,
        data: Sequence[float],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[float]: ...
    @overload
    def __new__(  # type: ignore[overload-cannot-match] # pyright: ignore[reportOverlappingOverload]
        cls,
        data: Sequence[int | float],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series[float]: ...
    @overload
    def __new__(
        cls,
        data: S1 | _ListLike[S1] | dict[HashableT1, S1] | KeysView[S1] | ValuesView[S1],
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        data: (
            Scalar
            | _ListLike
            | Mapping[HashableT1, Any]
            | BaseGroupBy
            | NaTType
            | NAType
            | None
        ) = ...,
        index: AxesData | None = ...,
        dtype: Dtype = ...,
        name: Hashable = ...,
        copy: bool = ...,
    ) -> Series: ...
    @property
    def hasnans(self) -> bool: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @property
    def dtypes(self) -> DtypeObj: ...
    @property
    def name(self) -> Hashable | None: ...
    @name.setter
    def name(self, value: Hashable | None) -> None: ...
    @property
    def values(self) -> ArrayLike: ...
    @property
    def array(self) -> ExtensionArray: ...
    def ravel(self, order: _str = ...) -> np.ndarray: ...
    def __len__(self) -> int: ...
    def view(self, dtype=...) -> Series[S1]: ...
    @final
    def __array_ufunc__(
        self, ufunc: Callable, method: _str, *inputs: Any, **kwargs: Any
    ): ...
    def __array__(
        self, dtype: _str | np.dtype = ..., copy: bool | None = ...
    ) -> np_1darray: ...
    @property
    def axes(self) -> list: ...
    @final
    def __getattr__(self, name: _str) -> S1: ...
    @overload
    def __getitem__(
        self,
        idx: (
            list[_str]
            | Index
            | Series[S1]
            | slice
            | MaskType
            | tuple[Hashable | slice, ...]
        ),
    ) -> Self: ...
    @overload
    def __getitem__(self, idx: Scalar) -> S1: ...
    def __setitem__(self, key, value) -> None: ...
    @overload
    def get(self, key: Hashable, default: None = ...) -> S1 | None: ...
    @overload
    def get(self, key: Hashable, default: S1) -> S1: ...
    @overload
    def get(self, key: Hashable, default: _T) -> S1 | _T: ...
    def repeat(
        self, repeats: int | list[int], axis: AxisIndex | None = 0
    ) -> Series[S1]: ...
    @property
    def index(self) -> Index | MultiIndex: ...
    @index.setter
    def index(self, idx: Index) -> None: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = ...,
        *,
        drop: Literal[False] = ...,
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = ...,
        *,
        drop: Literal[True],
        name: Level = ...,
        inplace: Literal[False] = ...,
        allow_duplicates: bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | Level | None = ...,
        *,
        drop: bool = ...,
        name: Level = ...,
        inplace: Literal[True],
        allow_duplicates: bool = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: FilePath | WriteBuffer[_str],
        na_rep: _str = ...,
        float_format: FloatFormatType = ...,
        header: _bool = ...,
        index: _bool = ...,
        length: _bool = ...,
        dtype: _bool = ...,
        name: _bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: None = ...,
        na_rep: _str = ...,
        float_format: FloatFormatType = ...,
        header: _bool = ...,
        index: _bool = ...,
        length: _bool = ...,
        dtype: _bool = ...,
        name: _bool = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[_str],
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: Literal[True],
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["a"],
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = ...,
        *,
        orient: Literal["records"],
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: Literal[True],
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["a"],
    ) -> _str: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePath | WriteBuffer[_str] | WriteBuffer[bytes],
        *,
        orient: JsonSeriesOrient | None = ...,
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: _bool = ...,
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["w"] = ...,
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: None = ...,
        *,
        orient: JsonSeriesOrient | None = ...,
        date_format: Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: TimeUnit = ...,
        default_handler: Callable[[Any], JSONSerializable] | None = ...,
        lines: _bool = ...,
        compression: CompressionOptions = ...,
        index: _bool = ...,
        indent: int | None = ...,
        mode: Literal["w"] = ...,
    ) -> _str: ...
    @final
    def to_xarray(self) -> xr.DataArray: ...
    def items(self) -> Iterator[tuple[Hashable, S1]]: ...
    def keys(self) -> Index: ...
    @overload
    def to_dict(self, *, into: type[dict] = ...) -> dict[Any, S1]: ...
    @overload
    def to_dict(
        self, *, into: type[MutableMapping] | MutableMapping
    ) -> MutableMapping[Hashable, S1]: ...
    def to_frame(self, name: object | None = ...) -> DataFrame: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: DatetimeIndex,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Timestamp]: ...
    @overload
    def groupby(
        self,
        by: TimedeltaIndex,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Timedelta]: ...
    @overload
    def groupby(
        self,
        by: PeriodIndex,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Period]: ...
    @overload
    def groupby(
        self,
        by: IntervalIndex[IntervalT],
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, IntervalT]: ...
    @overload
    def groupby(
        self,
        by: MultiIndex | GroupByObjectNonScalar,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, tuple]: ...
    @overload
    def groupby(
        self,
        by: None,
        level: IndexLabel,  # level is required when by=None (passed as positional)
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: None = ...,
        *,
        level: IndexLabel,  # level is required when by=None (passed as keyword)
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Scalar]: ...
    @overload
    def groupby(
        self,
        by: Series[SeriesByT],
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, SeriesByT]: ...
    @overload
    def groupby(
        self,
        by: CategoricalIndex | Index | Series,
        level: IndexLabel | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        observed: _bool | _NoDefaultDoNotUse = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy[S1, Any]: ...
    def count(self) -> int: ...
    def mode(self, dropna=True) -> Series[S1]: ...
    def unique(self) -> np.ndarray: ...
    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[True],
        ignore_index: _bool = ...,
    ) -> None: ...
    @overload
    def drop_duplicates(
        self,
        *,
        keep: DropKeep = ...,
        inplace: Literal[False] = ...,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    def duplicated(self, keep: DropKeep = "first") -> Series[_bool]: ...
    def idxmax(
        self,
        axis: AxisIndex = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int | _str: ...
    def idxmin(
        self,
        axis: AxisIndex = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> int | _str: ...
    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> Series[S1]: ...
    @overload
    def quantile(
        self,
        q: float = ...,
        interpolation: QuantileInterpolation = ...,
    ) -> float: ...
    @overload
    def quantile(
        self,
        q: _ListLike,
        interpolation: QuantileInterpolation = ...,
    ) -> Series[S1]: ...
    def corr(
        self,
        other: Series[S1],
        method: Literal["pearson", "kendall", "spearman"] = ...,
        min_periods: int | None = ...,
    ) -> float: ...
    def cov(
        self, other: Series[S1], min_periods: int | None = None, ddof: int = 1
    ) -> float: ...
    @overload
    def diff(self: Series[_bool], periods: int = ...) -> Series[type[object]]: ...  # type: ignore[overload-overlap]
    @overload
    def diff(self: Series[complex], periods: int = ...) -> Series[complex]: ...  # type: ignore[overload-overlap]
    @overload
    def diff(self: Series[bytes], periods: int = ...) -> Never: ...
    @overload
    def diff(self: Series[type], periods: int = ...) -> Never: ...
    @overload
    def diff(self: Series[_str], periods: int = ...) -> Never: ...
    @overload
    def diff(self, periods: int = ...) -> Series[float]: ...
    def autocorr(self, lag: int = 1) -> float: ...
    @overload
    def dot(self, other: Series[S1]) -> Scalar: ...
    @overload
    def dot(self, other: DataFrame) -> Series[S1]: ...
    @overload
    def dot(
        self, other: ArrayLike | dict[_str, np.ndarray] | Sequence[S1] | Index[S1]
    ) -> np.ndarray: ...
    @overload
    def __matmul__(self, other: Series) -> Scalar: ...
    @overload
    def __matmul__(self, other: DataFrame) -> Series: ...
    @overload
    def __matmul__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def __rmatmul__(self, other: Series) -> Scalar: ...
    @overload
    def __rmatmul__(self, other: DataFrame) -> Series: ...
    @overload
    def __rmatmul__(self, other: np.ndarray) -> np.ndarray: ...
    @overload
    def searchsorted(
        self,
        value: _ListLike,
        side: Literal["left", "right"] = ...,
        sorter: _ListLike | None = ...,
    ) -> list[int]: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Literal["left", "right"] = ...,
        sorter: _ListLike | None = ...,
    ) -> int: ...
    @overload
    def compare(
        self,
        other: Series,
        align_axis: AxisIndex,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
        result_names: Suffixes = ...,
    ) -> Series: ...
    @overload
    def compare(
        self,
        other: Series,
        align_axis: AxisColumn = ...,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
        result_names: Suffixes = ...,
    ) -> DataFrame: ...
    def combine(
        self, other: Series[S1], func: Callable, fill_value: Scalar | None = ...
    ) -> Series[S1]: ...
    def combine_first(self, other: Series[S1]) -> Series[S1]: ...
    def update(self, other: Series[S1] | Sequence[S1] | Mapping[int, S1]) -> None: ...
    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: _bool = ...,
        inplace: Literal[True],
        key: ValueKeyFunc = ...,
    ) -> None: ...
    @overload
    def sort_values(
        self,
        *,
        axis: Axis = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        ignore_index: _bool = ...,
        inplace: Literal[False] = ...,
        key: ValueKeyFunc = ...,
    ) -> Series[S1]: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: Level | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        inplace: Literal[True],
        key: IndexKeyFunc = ...,
    ) -> None: ...
    @overload
    def sort_index(
        self,
        *,
        axis: Axis = ...,
        level: Level | list[int] | list[_str] | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: SortKind = ...,
        na_position: NaPosition = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        inplace: Literal[False] = ...,
        key: IndexKeyFunc = ...,
    ) -> Series[S1]: ...
    def argsort(
        self,
        axis: AxisIndex = 0,
        kind: SortKind = "quicksort",
        order: None = None,
        stable: None = None,
    ) -> Series[int]: ...
    def nlargest(
        self, n: int = 5, keep: NsmallestNlargestKeep = "first"
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = 5, keep: NsmallestNlargestKeep = "first"
    ) -> Series[S1]: ...
    def swaplevel(
        self, i: Level = -2, j: Level = -1, copy: _bool = True
    ) -> Series[S1]: ...
    def reorder_levels(self, order: list) -> Series[S1]: ...
    def explode(self, ignore_index: _bool = ...) -> Series[S1]: ...
    def unstack(
        self,
        level: IndexLabel = -1,
        fill_value: int | _str | dict | None = None,
        sort: _bool = True,
    ) -> DataFrame: ...
    @overload
    def map(
        self,
        arg: Callable[[S1], S2 | NAType] | Mapping[S1, S2] | Series[S2],
        na_action: Literal["ignore"] = ...,
    ) -> Series[S2]: ...
    @overload
    def map(
        self,
        arg: Callable[[S1 | NAType], S2 | NAType] | Mapping[S1, S2] | Series[S2],
        na_action: None = ...,
    ) -> Series[S2]: ...
    @overload
    def map(
        self,
        arg: Callable[[Any], Any] | Mapping[Any, Any] | Series,
        na_action: Literal["ignore"] | None = ...,
    ) -> Series: ...
    @overload
    def aggregate(
        self: Series[int],
        func: Literal["mean"],
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeBase,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> S1: ...
    @overload
    def aggregate(
        self,
        func: AggFuncTypeSeriesToFrame = ...,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series: ...
    agg = aggregate
    @overload
    def transform(
        self,
        func: AggFuncTypeBase,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @overload
    def transform(
        self,
        func: list[AggFuncTypeBase] | AggFuncTypeDictFrame,
        axis: AxisIndex = ...,
        *args: Any,
        **kwargs: Any,
    ) -> DataFrame: ...
    @overload
    def apply(
        self,
        func: Callable[
            ..., Scalar | Sequence | set | Mapping | NAType | frozenset | None
        ],
        convertDType: _bool = ...,
        args: tuple = ...,
        **kwargs: Any,
    ) -> Series: ...
    @overload
    def apply(
        self,
        func: Callable[..., BaseOffset],
        convertDType: _bool = ...,
        args: tuple = ...,
        **kwargs: Any,
    ) -> OffsetSeries: ...
    @overload
    def apply(
        self,
        func: Callable[..., Series],
        convertDType: _bool = ...,
        args: tuple = ...,
        **kwargs: Any,
    ) -> DataFrame: ...
    @final
    def align(
        self,
        other: DataFrame | Series,
        join: JoinHow = "outer",
        axis: Axis | None = 0,
        level: Level | None = None,
        copy: _bool = True,
        fill_value: Scalar | NAType | None = None,
    ) -> tuple[Series, Series]: ...
    @overload
    def rename(
        self,
        index: Callable[[Any], Label],
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Mapping[Any, Label],
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Scalar | tuple[Hashable, ...] | None = None,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @overload
    def rename(
        self,
        index: Renamer | Scalar | tuple[Hashable, ...] | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Self: ...
    @final
    def reindex_like(
        self,
        other: Series[S1],
        method: FillnaOptions | Literal["nearest"] | None = None,
        copy: _bool = True,
        limit: int | None = None,
        tolerance: Scalar | AnyArrayLike | Sequence[Scalar] | None = None,
    ) -> Self: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict | Series[S1] | DataFrame | None = ...,
        *,
        axis: AxisIndex = ...,
        limit: int | None = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | NAType | dict | Series[S1] | DataFrame | None = ...,
        *,
        axis: AxisIndex = ...,
        limit: int | None = ...,
        inplace: Literal[False] = ...,
    ) -> Series[S1]: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue = ...,
        value: ReplaceValue = ...,
        *,
        regex: ReplaceValue = ...,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace: ReplaceValue = ...,
        value: ReplaceValue = ...,
        *,
        regex: ReplaceValue = ...,
        inplace: Literal[False] = ...,
    ) -> Series[S1]: ...
    def shift(
        self,
        periods: int | Sequence[int] = ...,
        freq: BaseOffset | timedelta | _str | None = None,
        axis: Axis = 0,
        fill_value: Scalar | NAType | None = ...,
    ) -> Series: ...
    def info(
        self,
        verbose: bool | None = ...,
        buf: WriteBuffer[_str] | None = ...,
        memory_usage: bool | Literal["deep"] | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...
    def memory_usage(self, index: _bool = True, deep: _bool = False) -> int: ...
    def isin(self, values: Iterable | Series[S1] | dict) -> Series[_bool]: ...
    def between(
        self,
        left: Scalar | ListLikeU,
        right: Scalar | ListLikeU,
        inclusive: Literal["both", "neither", "left", "right"] = ...,
    ) -> Series[_bool]: ...
    def isna(self) -> Series[_bool]: ...
    def isnull(self) -> Series[_bool]: ...
    def notna(self) -> Series[_bool]: ...
    def notnull(self) -> Series[_bool]: ...
    @overload
    def dropna(
        self,
        *,
        axis: AxisIndex = ...,
        inplace: Literal[True],
        how: AnyAll | None = ...,
        ignore_index: _bool = ...,
    ) -> None: ...
    @overload
    def dropna(
        self,
        *,
        axis: AxisIndex = ...,
        inplace: Literal[False] = ...,
        how: AnyAll | None = ...,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    def to_timestamp(
        self,
        freq=...,
        how: ToTimestampHow = "start",
        copy: _bool = True,
    ) -> Series[S1]: ...
    def to_period(self, freq: _str | None = None, copy: _bool = True) -> DataFrame: ...
    @property
    def str(
        self,
    ) -> StringMethods[
        Self,
        DataFrame,
        Series[bool],
        Series[list[_str]],
        Series[int],
        Series[bytes],
        Series[_str],
        Series,
    ]: ...
    @property
    def dt(self) -> CombinedDatetimelikeProperties: ...
    @property
    def plot(self) -> PlotAccessor: ...
    sparse = ...
    def hist(
        self,
        by: object | None = None,
        ax: PlotAxes | None = None,
        grid: _bool = True,
        xlabelsize: float | _str | None = None,
        xrot: float | None = None,
        ylabelsize: float | _str | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence = 10,
        backend: _str | None = None,
        legend: _bool = False,
        **kwargs: Any,
    ) -> SubplotBase: ...
    @final
    def swapaxes(
        self, axis1: AxisIndex, axis2: AxisIndex, copy: _bool = ...
    ) -> Series[S1]: ...
    @final
    def droplevel(self, level: Level | list[Level], axis: AxisIndex = 0) -> Self: ...
    def pop(self, item: Hashable) -> S1: ...
    @final
    def squeeze(self, axis: None = None) -> Series[S1] | Scalar: ...
    @final
    def __abs__(self) -> Series[S1]: ...
    @final
    def add_prefix(self, prefix: _str, axis: AxisIndex | None = None) -> Series[S1]: ...
    @final
    def add_suffix(self, suffix: _str, axis: AxisIndex | None = None) -> Series[S1]: ...
    def reindex(
        self,
        index: Axes | None = None,
        method: ReindexMethod | None = None,
        copy: bool = True,
        level: int | _str | None = None,
        fill_value: Scalar | None = None,
        limit: int | None = None,
        tolerance: float | Timedelta | None = None,
    ) -> Series[S1]: ...
    def filter(
        self,
        items: _ListLike | None = None,
        like: _str | None = None,
        regex: _str | None = None,
        axis: AxisIndex | None = None,
    ) -> Series[S1]: ...
    @final
    def head(self, n: int = 5) -> Series[S1]: ...
    @final
    def tail(self, n: int = 5) -> Series[S1]: ...
    @final
    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: _bool = False,
        weights: _str | _ListLike | np.ndarray | None = None,
        random_state: RandomState | None = None,
        axis: AxisIndex | None = None,
        ignore_index: _bool = False,
    ) -> Series[S1]: ...
    @overload
    def astype(
        self,
        dtype: BooleanDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[bool]: ...
    @overload
    def astype(
        self,
        dtype: IntDtypeArg | UIntDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[int]: ...
    @overload
    def astype(
        self,
        dtype: StrDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[_str]: ...
    @overload
    def astype(
        self,
        dtype: BytesDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[bytes]: ...
    @overload
    def astype(
        self,
        dtype: FloatDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[float]: ...
    @overload
    def astype(
        self,
        dtype: ComplexDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[complex]: ...
    @overload
    def astype(
        self,
        dtype: TimedeltaDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> TimedeltaSeries: ...
    @overload
    def astype(
        self,
        dtype: TimestampDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> TimestampSeries: ...
    @overload
    def astype(
        self,
        dtype: CategoryDtypeArg,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series[CategoricalDtype]: ...
    @overload
    def astype(
        self,
        dtype: ObjectDtypeArg | VoidDtypeArg | ExtensionDtype | DtypeObj,
        copy: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @final
    def copy(self, deep: _bool = True) -> Series[S1]: ...
    @final
    def infer_objects(self, copy: _bool = True) -> Series[S1]: ...
    @overload
    def ffill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
    ) -> Series[S1]: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: AxisIndex | None = 0,
        limit: int | None = ...,
        inplace: Literal[True],
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: InterpolateOptions = ...,
        *,
        axis: AxisIndex | None = 0,
        limit: int | None = ...,
        inplace: Literal[False] = ...,
        limit_direction: Literal["forward", "backward", "both"] | None = ...,
        limit_area: Literal["inside", "outside"] | None = ...,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @final
    def asof(
        self,
        where: Scalar | Sequence[Scalar],
        subset: _str | Sequence[_str] | None = None,
    ) -> Scalar | Series[S1]: ...
    @overload
    def clip(  # pyright: ignore[reportOverlappingOverload]
        self,
        lower: None = ...,
        upper: None = ...,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> Self: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | float | None = ...,
        upper: AnyArrayLike | float | None = ...,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[True],
        **kwargs: Any,
    ) -> None: ...
    @overload
    def clip(
        self,
        lower: AnyArrayLike | float | None = ...,
        upper: AnyArrayLike | float | None = ...,
        *,
        axis: AxisIndex | None = 0,
        inplace: Literal[False] = ...,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @final
    def asfreq(
        self,
        freq: DateOffset | _str,
        method: FillnaOptions | None = None,
        how: Literal["start", "end"] | None = None,
        normalize: _bool = False,
        fill_value: Scalar | None = None,
    ) -> Series[S1]: ...
    @final
    def at_time(
        self,
        time: _str | time,
        asof: _bool = False,
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    @final
    def between_time(
        self,
        start_time: _str | time,
        end_time: _str | time,
        inclusive: IntervalClosedType = "both",
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    @final
    def first(self, offset) -> Series[S1]: ...
    @final
    def last(self, offset) -> Series[S1]: ...
    @final
    def rank(
        self,
        axis: AxisIndex = 0,
        method: Literal["average", "min", "max", "first", "dense"] = "average",
        numeric_only: _bool = False,
        na_option: Literal["keep", "top", "bottom"] = "keep",
        ascending: _bool = True,
        pct: _bool = False,
    ) -> Series[float]: ...
    @overload
    def where(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other=...,
        *,
        inplace: Literal[True],
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> None: ...
    @overload
    def where(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other=...,
        *,
        inplace: Literal[False] = ...,
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> Self: ...
    @overload
    def mask(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Series[S1] | DataFrame | Callable | NAType | None = ...,
        *,
        inplace: Literal[True],
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> None: ...
    @overload
    def mask(
        self,
        cond: (
            Series[S1]
            | Series[_bool]
            | np.ndarray
            | Callable[[Series[S1]], Series[bool]]
            | Callable[[S1], bool]
        ),
        other: Scalar | Series[S1] | DataFrame | Callable | NAType | None = ...,
        *,
        inplace: Literal[False] = ...,
        axis: AxisIndex | None = 0,
        level: Level | None = ...,
    ) -> Series[S1]: ...
    def case_when(
        self,
        caselist: list[
            tuple[
                Sequence[bool]
                | Series[bool]
                | Callable[[Series], Series | np.ndarray | Sequence[bool]],
                ListLikeU | Scalar | Callable[[Series], Series | np.ndarray],
            ],
        ],
    ) -> Series: ...
    @final
    def truncate(
        self,
        before: date | _str | int | None = ...,
        after: date | _str | int | None = ...,
        axis: AxisIndex | None = 0,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    @final
    def tz_convert(
        self,
        tz: TimeZones,
        axis: AxisIndex = 0,
        level: Level | None = None,
        copy: _bool = True,
    ) -> Series[S1]: ...
    @final
    def tz_localize(
        self,
        tz: TimeZones,
        axis: AxisIndex = 0,
        level: Level | None = None,
        copy: _bool = True,
        ambiguous: TimeAmbiguous = "raise",
        nonexistent: _str = "raise",
    ) -> Series[S1]: ...
    @final
    def abs(self) -> Series[S1]: ...
    @final
    def describe(
        self,
        percentiles: list[float] | None = ...,
        include: Literal["all"] | list[S1] | None = ...,
        exclude: S1 | list[S1] | None = ...,
    ) -> Series[S1]: ...
    @final
    def pct_change(
        self,
        periods: int = 1,
        fill_method: None = None,
        freq: DateOffset | timedelta | _str | None = None,
    ) -> Series[float]: ...
    @final
    def first_valid_index(self) -> Scalar: ...
    @final
    def last_valid_index(self) -> Scalar: ...
    @overload
    def value_counts(  # pyrefly: ignore
        self,
        normalize: Literal[False] = ...,
        sort: _bool = ...,
        ascending: _bool = ...,
        bins: int | None = ...,
        dropna: _bool = ...,
    ) -> Series[int]: ...
    @overload
    def value_counts(
        self,
        normalize: Literal[True],
        sort: _bool = ...,
        ascending: _bool = ...,
        bins: int | None = ...,
        dropna: _bool = ...,
    ) -> Series[float]: ...
    @final
    @property
    def T(self) -> Self: ...
    # The rest of these were left over from the old
    # stubs we shipped in preview. They may belong in
    # the base classes in some cases; I expect stubgen
    # just failed to generate these so I couldn't match
    # them up.
    @overload
    def __add__(self: Series[Never], other: Scalar | _ListLike | Series) -> Series: ...
    @overload
    def __add__(self, other: Series[Never]) -> Series: ...
    @overload
    def __add__(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __add__(self: Series[bool], other: np_ndarray_bool) -> Series[bool]: ...
    @overload
    def __add__(self: Series[bool], other: np_ndarray_anyint) -> Series[int]: ...
    @overload
    def __add__(self: Series[bool], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __add__(self: Series[bool], other: np_ndarray_complex) -> Series[complex]: ...
    @overload
    def __add__(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
    ) -> Series[int]: ...
    @overload
    def __add__(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __add__(self: Series[int], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __add__(self: Series[int], other: np_ndarray_complex) -> Series[complex]: ...
    @overload
    def __add__(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __add__(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __add__(self: Series[float], other: np_ndarray_complex) -> Series[complex]: ...
    @overload
    def __add__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | np_ndarray_complex
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __add__(
        self: Series[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __add__(
        self: Series[_str], other: _str | Sequence[_str] | np_ndarray_str | Series[_str]
    ) -> Series[_str]: ...
    @overload
    def add(
        self: Series[Never],
        other: Scalar | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def add(
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def add(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def add(
        self: Series[bool],
        other: np_ndarray_bool,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[bool]: ...
    @overload
    def add(
        self: Series[bool],
        other: np_ndarray_anyint,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def add(
        self: Series[bool],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def add(
        self: Series[bool],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def add(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def add(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def add(
        self: Series[int],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def add(
        self: Series[int],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def add(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def add(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def add(
        self: Series[float],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def add(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | np_ndarray_complex
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def add(
        self: Series[_str],
        other: _str | Sequence[_str] | np_ndarray_str | Series[_str],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_str]: ...
    @overload  # type: ignore[override]
    def __radd__(self: Series[Never], other: Scalar | _ListLike) -> Series: ...
    @overload
    def __radd__(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __radd__(self: Series[bool], other: np_ndarray_bool) -> Series[bool]: ...
    @overload
    def __radd__(self: Series[bool], other: np_ndarray_anyint) -> Series[int]: ...
    @overload
    def __radd__(self: Series[bool], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __radd__(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
    ) -> Series[int]: ...
    @overload
    def __radd__(
        self: Series[int], other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX]
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __radd__(self: Series[int], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __radd__(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __radd__(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __radd__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __radd__(
        self: Series[_T_COMPLEX], other: np_ndarray_complex
    ) -> Series[complex]: ...
    @overload
    def __radd__(
        self: Series[_str],
        other: (
            np_ndarray_bool | np_ndarray_anyint | np_ndarray_float | np_ndarray_complex
        ),
    ) -> Never: ...
    @overload
    def __radd__(
        self: Series[_str], other: _str | Sequence[_str] | np_ndarray_str | Series[_str]
    ) -> Series[_str]: ...
    @overload
    def radd(
        self: Series[Never],
        other: Scalar | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def radd(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def radd(
        self: Series[bool],
        other: np_ndarray_bool,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[bool]: ...
    @overload
    def radd(
        self: Series[bool],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def radd(
        self: Series[bool],
        other: np_ndarray_anyint,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def radd(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def radd(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def radd(
        self: Series[int],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def radd(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def radd(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def radd(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def radd(
        self: Series[_T_COMPLEX],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def radd(
        self: Series[_str],
        other: _str | Sequence[_str] | np_ndarray_str | Series[_str],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_str]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __and__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | list[int] | MaskType
    ) -> Series[bool]: ...
    @overload
    def __and__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __eq__(self, other: object) -> Series[_bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __floordiv__(self, other: num | _ListLike | Series[S1]) -> Series[int]: ...
    def __ge__(  # type: ignore[override]
        self, other: S1 | _ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __gt__(  # type: ignore[override]
        self, other: S1 | _ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __le__(  # type: ignore[override]
        self, other: S1 | _ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    def __lt__(  # type: ignore[override]
        self, other: S1 | _ListLike | Series[S1] | datetime | timedelta | date
    ) -> Series[_bool]: ...
    @overload
    def __mul__(self: Series[Never], other: complex | _ListLike | Series) -> Series: ...
    @overload
    def __mul__(self, other: Series[Never]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __mul__(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __mul__(self: Series[bool], other: np_ndarray_bool) -> Series[bool]: ...
    @overload
    def __mul__(self: Series[bool], other: np_ndarray_anyint) -> Series[int]: ...
    @overload
    def __mul__(self: Series[bool], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __mul__(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
    ) -> Series[int]: ...
    @overload
    def __mul__(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __mul__(self: Series[int], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __mul__(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __mul__(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __mul__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __mul__(
        self: Series[_T_COMPLEX], other: np_ndarray_complex
    ) -> Series[complex]: ...
    @overload
    def __mul__(
        self, other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64
    ) -> TimedeltaSeries: ...
    @overload
    def mul(
        self: Series[Never],
        other: complex | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def mul(  # type: ignore[overload-overlap]
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def mul(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def mul(
        self: Series[bool],
        other: np_ndarray_bool,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[bool]: ...
    @overload
    def mul(
        self: Series[bool],
        other: np_ndarray_anyint,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def mul(
        self: Series[bool],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def mul(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def mul(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def mul(
        self: Series[int],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def mul(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def mul(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def mul(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def mul(
        self: Series[_T_COMPLEX],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def mul(
        self,
        other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> TimedeltaSeries: ...
    @overload
    def __rmul__(
        self: Series[Never], other: complex | _ListLike | Series
    ) -> Series: ...
    @overload
    def __rmul__(self, other: Series[Never]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __rmul__(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __rmul__(self: Series[bool], other: np_ndarray_bool) -> Series[bool]: ...
    @overload
    def __rmul__(self: Series[bool], other: np_ndarray_anyint) -> Series[int]: ...
    @overload
    def __rmul__(self: Series[bool], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __rmul__(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
    ) -> Series[int]: ...
    @overload
    def __rmul__(
        self: Series[int], other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX]
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __rmul__(self: Series[int], other: np_ndarray_float) -> Series[float]: ...
    @overload
    def __rmul__(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __rmul__(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __rmul__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __rmul__(
        self: Series[_T_COMPLEX], other: np_ndarray_complex
    ) -> Series[complex]: ...
    @overload
    def __rmul__(
        self, other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64
    ) -> TimedeltaSeries: ...
    @overload
    def rmul(
        self: Series[Never],
        other: complex | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rmul(  # type: ignore[overload-overlap]
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rmul(
        self: Series[bool],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def rmul(
        self: Series[bool],
        other: np_ndarray_bool,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[bool]: ...
    @overload
    def rmul(
        self: Series[bool],
        other: np_ndarray_anyint,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rmul(
        self: Series[bool],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rmul(
        self: Series[int],
        other: (
            bool | Sequence[bool] | np_ndarray_bool | np_ndarray_anyint | Series[bool]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rmul(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def rmul(
        self: Series[int],
        other: np_ndarray_float,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rmul(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rmul(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def rmul(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def rmul(
        self: Series[_T_COMPLEX],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def rmul(
        self,
        other: timedelta | Timedelta | TimedeltaSeries | np.timedelta64,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> TimedeltaSeries: ...
    def __mod__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __ne__(self, other: object) -> Series[_bool]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __pow__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __or__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | list[int] | MaskType
    ) -> Series[bool]: ...
    @overload
    def __or__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __rand__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __rand__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    def __rdivmod__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __rfloordiv__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rmod__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rpow__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __ror__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __ror__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __rxor__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __rxor__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[Never],
        other: datetime | np.datetime64 | np_ndarray_dt | TimestampSeries,
    ) -> TimedeltaSeries: ...
    @overload
    def __sub__(self: Series[Never], other: complex | _ListLike | Series) -> Series: ...
    @overload
    def __sub__(self, other: Series[Never]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __sub__(
        self: Series[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Series[int],
    ) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
    ) -> Series[float]: ...
    @overload
    def __sub__(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Series[bool]
            | Series[int]
        ),
    ) -> Series[int]: ...
    @overload
    def __sub__(
        self: Series[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
    ) -> Series[float]: ...
    @overload
    def __sub__(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[bool]
            | Series[int]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __sub__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __sub__(
        self: Series[_T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Series[complex]
        ),
    ) -> Series[complex]: ...
    @overload
    def __sub__(
        self: Series[Timestamp],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaSeries
            | TimedeltaIndex
        ),
    ) -> TimestampSeries: ...
    @overload
    def __sub__(
        self: Series[Timedelta],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaSeries
            | TimedeltaIndex
        ),
    ) -> TimedeltaSeries: ...
    @overload
    def sub(
        self: Series[Never],
        other: datetime | np.datetime64 | np_ndarray_dt | TimestampSeries,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> TimedeltaSeries: ...
    @overload
    def sub(
        self: Series[Never],
        other: complex | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def sub(  # type: ignore[overload-overlap]
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def sub(
        self: Series[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def sub(
        self: Series[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def sub(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Series[bool]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def sub(
        self: Series[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def sub(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[bool]
            | Series[int]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def sub(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def sub(
        self: Series[_T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Series[complex]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def sub(
        self: Series[Timestamp],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaSeries
            | TimedeltaIndex
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> TimestampSeries: ...
    @overload
    def sub(
        self: Series[Timedelta],
        other: (
            timedelta
            | np.timedelta64
            | np_ndarray_td
            | TimedeltaSeries
            | TimedeltaIndex
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> TimedeltaSeries: ...
    @overload
    def __rsub__(  # type: ignore[misc]
        self: Series[Never],
        other: datetime | np.datetime64 | np_ndarray_dt | TimestampSeries,
    ) -> TimedeltaSeries: ...
    @overload
    def __rsub__(
        self: Series[Never], other: complex | _ListLike | Series
    ) -> Series: ...
    @overload
    def __rsub__(self, other: Series[Never]) -> Series: ...
    @overload
    def __rsub__(
        self: Series[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Series[int],
    ) -> Series[int]: ...
    @overload
    def __rsub__(
        self: Series[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
    ) -> Series[float]: ...
    @overload
    def __rsub__(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Series[bool]
            | Series[int]
        ),
    ) -> Series[int]: ...
    @overload
    def __rsub__(
        self: Series[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
    ) -> Series[float]: ...
    @overload
    def __rsub__(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[bool]
            | Series[int]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __rsub__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __rsub__(
        self: Series[_T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Series[complex]
        ),
    ) -> Series[complex]: ...
    @overload
    def rsub(
        self: Series[Never],
        other: datetime | np.datetime64 | np_ndarray_dt | TimestampSeries,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> TimedeltaSeries: ...
    @overload
    def rsub(
        self: Series[Never],
        other: complex | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rsub(
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series: ...
    @overload
    def rsub(
        self: Series[bool],
        other: Just[int] | Sequence[Just[int]] | np_ndarray_anyint | Series[int],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rsub(
        self: Series[bool],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rsub(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | Series[bool]
            | Series[int]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[int]: ...
    @overload
    def rsub(
        self: Series[int],
        other: Just[float] | Sequence[Just[float]] | np_ndarray_float | Series[float],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rsub(
        self: Series[float],
        other: (
            float
            | Sequence[float]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[bool]
            | Series[int]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[float]: ...
    @overload
    def rsub(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def rsub(
        self: Series[_T_COMPLEX],
        other: (
            Just[complex]
            | Sequence[Just[complex]]
            | np_ndarray_complex
            | Series[complex]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: int = 0,
    ) -> Series[complex]: ...
    @overload
    def __truediv__(
        self: Series[Never], other: complex | _ListLike | Series
    ) -> Series: ...
    @overload
    def __truediv__(self, other: Series[Never]) -> Series: ...
    @overload
    def __truediv__(self: Series[bool], other: np_ndarray_bool) -> Never: ...
    @overload
    def __truediv__(
        self: Series[bool],
        other: (
            Just[int]
            | Just[float]
            | Sequence[float]
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[int]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[bool],
        other: Just[complex] | Sequence[Just[complex]] | Series[complex],
    ) -> Series[complex]: ...
    @overload
    def __truediv__(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __truediv__(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __truediv__(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __truediv__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __truediv__(
        self: Series[_T_COMPLEX], other: np_ndarray_complex
    ) -> Series[complex]: ...
    @overload
    def __truediv__(self, other: Path) -> Series: ...
    @overload
    def truediv(
        self: Series[Never],
        other: complex | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def truediv(
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def truediv(
        self: Series[bool],
        other: (
            Just[int]
            | Just[float]
            | Sequence[float]
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[int]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[bool],
        other: Just[complex] | Sequence[Just[complex]] | Series[complex],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def truediv(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def truediv(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def truediv(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def truediv(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def truediv(
        self: Series[_T_COMPLEX],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def truediv(
        self,
        other: Path,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    div = truediv
    @overload
    def __rtruediv__(
        self: Series[Never], other: complex | _ListLike | Series
    ) -> Series: ...
    @overload
    def __rtruediv__(self, other: Series[Never]) -> Series: ...
    @overload
    def __rtruediv__(self: Series[bool], other: np_ndarray_bool) -> Never: ...
    @overload
    def __rtruediv__(
        self: Series[bool],
        other: (
            Just[int]
            | Just[float]
            | Sequence[float]
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[int]
            | Series[float]
        ),
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[bool],
        other: Just[complex] | Sequence[Just[complex]] | Series[complex],
    ) -> Series[complex]: ...
    @overload
    def __rtruediv__(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[int], other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX]
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __rtruediv__(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def __rtruediv__(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
    ) -> Series[complex]: ...
    @overload
    def __rtruediv__(
        self: Series[_T_COMPLEX], other: np_ndarray_complex
    ) -> Series[complex]: ...
    @overload
    def __rtruediv__(self, other: Path) -> Series: ...
    @overload
    def rtruediv(
        self: Series[Never],
        other: complex | _ListLike | Series,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def rtruediv(
        self,
        other: Series[Never],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    @overload
    def rtruediv(
        self: Series[bool],
        other: (
            Just[int]
            | Just[float]
            | Sequence[float]
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[int]
            | Series[float]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[bool],
        other: Just[complex] | Sequence[Just[complex]] | Series[complex],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def rtruediv(
        self: Series[int],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[int],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def rtruediv(
        self: Series[float],
        other: (
            int
            | Sequence[int]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_INT]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[float]: ...
    @overload
    def rtruediv(
        self: Series[float],
        other: _T_COMPLEX | Sequence[_T_COMPLEX] | Series[_T_COMPLEX],
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[_T_COMPLEX]: ...
    @overload
    def rtruediv(
        self: Series[complex],
        other: (
            _T_COMPLEX
            | Sequence[_T_COMPLEX]
            | np_ndarray_bool
            | np_ndarray_anyint
            | np_ndarray_float
            | Series[_T_COMPLEX]
        ),
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def rtruediv(
        self: Series[_T_COMPLEX],
        other: np_ndarray_complex,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series[complex]: ...
    @overload
    def rtruediv(
        self,
        other: Path,
        level: Level | None = None,
        fill_value: float | None = None,
        axis: AxisIndex = 0,
    ) -> Series: ...
    rdiv = rtruediv
    # ignore needed for mypy as we want different results based on the arguments
    @overload  # type: ignore[override]
    def __xor__(  # pyright: ignore[reportOverlappingOverload]
        self, other: bool | MaskType | list[int]
    ) -> Series[bool]: ...
    @overload
    def __xor__(self, other: int | np_ndarray_anyint | Series[int]) -> Series[int]: ...
    @final
    def __invert__(self) -> Series[bool]: ...
    # properties
    # @property
    # def array(self) -> _npndarray
    @property
    def at(self) -> _AtIndexer: ...
    @property
    def cat(self) -> CategoricalAccessor: ...
    @property
    def iat(self) -> _iAtIndexer: ...
    @property
    def iloc(self) -> _iLocIndexerSeries[S1]: ...
    @property
    def loc(self) -> _LocIndexerSeries[S1]: ...
    def all(
        self,
        axis: AxisIndex = 0,
        bool_only: _bool | None = False,
        skipna: _bool = True,
        **kwargs: Any,
    ) -> np.bool: ...
    def any(
        self,
        *,
        axis: AxisIndex = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        **kwargs: Any,
    ) -> np.bool: ...
    def cummax(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cummin(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    @overload
    def cumprod(
        self: Series[_str],
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Never: ...
    @overload
    def cumprod(
        self,
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def cumsum(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Series[S1]: ...
    def divmod(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def eq(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    @final
    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        min_periods: int = 0,
        adjust: _bool = True,
        ignore_na: _bool = False,
        axis: Axis = 0,
        times: np.ndarray | Series | None = None,
        method: CalculationMethod = "single",
    ) -> ExponentialMovingWindow[Series]: ...
    @final
    def expanding(
        self,
        min_periods: int = 1,
        axis: Literal[0] = 0,
        method: CalculationMethod = "single",
    ) -> Expanding[Series]: ...
    def floordiv(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[int]: ...
    def ge(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    def gt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    @final
    def item(self) -> S1: ...
    def kurt(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def kurtosis(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def le(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    def lt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    def max(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = ...,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S1: ...
    def mean(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = ...,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    def median(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = ...,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    def min(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = True,
        level: None = ...,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> S1: ...
    def mod(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    def ne(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[_bool]: ...
    @final
    def nunique(self, dropna: _bool = True) -> int: ...
    def pow(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex | None = 0,
    ) -> Series[S1]: ...
    def prod(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Scalar: ...
    def product(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        min_count: int = 0,
        **kwargs: Any,
    ) -> Scalar: ...
    def rdivmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def rfloordiv(
        self,
        other,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def rmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    @overload
    def rolling(
        self,
        window: int | _str | timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = ...,
        center: _bool = ...,
        on: _str | None = ...,
        closed: IntervalClosedType | None = ...,
        step: int | None = ...,
        method: CalculationMethod = ...,
        *,
        win_type: _str,
    ) -> Window[Series]: ...
    @overload
    def rolling(
        self,
        window: int | _str | timedelta | BaseOffset | BaseIndexer,
        min_periods: int | None = ...,
        center: _bool = ...,
        on: _str | None = ...,
        closed: IntervalClosedType | None = ...,
        step: int | None = ...,
        method: CalculationMethod = ...,
        *,
        win_type: None = ...,
    ) -> Rolling[Series]: ...
    def rpow(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = None,
        axis: AxisIndex = ...,
    ) -> Series[S1]: ...
    def sem(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def skew(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    def std(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> float: ...
    @overload
    def sum(
        self: Series[Never],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = ...,
        numeric_only: _bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> Any: ...
    # between `Series[bool]` and `Series[int]`.
    @overload
    def sum(
        self: Series[bool],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = ...,
        numeric_only: _bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> int: ...
    @overload
    def sum(
        self: Series[S1],
        axis: AxisIndex | None = 0,
        skipna: _bool | None = ...,
        numeric_only: _bool = ...,
        min_count: int = ...,
        **kwargs: Any,
    ) -> S1: ...
    def to_list(self) -> list[S1]: ...
    def tolist(self) -> list[S1]: ...
    def var(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = True,
        ddof: int = 1,
        numeric_only: _bool = False,
        **kwargs: Any,
    ) -> Scalar: ...
    # Rename axis with `mapper`, `axis`, and `inplace=True`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: AxisIndex | None = 0,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `mapper`, `axis`, and `inplace=False`
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike | None = ...,
        *,
        axis: AxisIndex | None = 0,
        copy: _bool = ...,
        inplace: Literal[False] = ...,
    ) -> Self: ...
    # Rename axis with `index` and `inplace=True`
    @overload
    def rename_axis(
        self,
        *,
        index: Scalar | ListLike | Callable | dict | None = ...,
        copy: _bool = ...,
        inplace: Literal[True],
    ) -> None: ...
    # Rename axis with `index` and `inplace=False`
    @overload
    def rename_axis(
        self,
        *,
        index: Scalar | ListLike | Callable | dict | None = ...,
        copy: _bool = ...,
        inplace: Literal[False] = ...,
    ) -> Self: ...
    def set_axis(self, labels, *, axis: Axis = ..., copy: _bool = ...) -> Self: ...
    def __iter__(self) -> Iterator[S1]: ...
    @final
    def xs(
        self,
        key: Hashable,
        axis: AxisIndex = 0,
        level: Level | None = ...,
        drop_level: _bool = True,
    ) -> Self: ...
    @final
    def __bool__(self) -> NoReturn: ...

@type_check_only
class _SeriesSubclassBase(Series[S1], Generic[S1, GenericT_co]):
    @overload
    def to_numpy(  # pyrefly: ignore
        self,
        dtype: None = None,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs,
    ) -> np_1darray[GenericT_co]: ...
    @overload
    def to_numpy(
        self,
        dtype: np.dtype[GenericT] | SupportsDType[GenericT] | type[GenericT],
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs,
    ) -> np_1darray[GenericT]: ...
    @overload
    def to_numpy(
        self,
        dtype: DTypeLike,
        copy: bool = False,
        na_value: Scalar = ...,
        **kwargs,
    ) -> np_1darray: ...

class TimestampSeries(_SeriesSubclassBase[Timestamp, np.datetime64]):
    @property
    def dt(self) -> TimestampProperties: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __add__(self, other: TimedeltaSeries | np.timedelta64 | timedelta | BaseOffset) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __radd__(self, other: TimedeltaSeries | np.timedelta64 | timedelta) -> TimestampSeries: ...  # type: ignore[override]
    @overload  # type: ignore[override]
    def __sub__(
        self, other: Timestamp | datetime | TimestampSeries
    ) -> TimedeltaSeries: ...
    @overload
    def __sub__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta | TimedeltaSeries | TimedeltaIndex | np.timedelta64 | BaseOffset
        ),
    ) -> TimestampSeries: ...
    def __mul__(self, other: float | Series[int] | Series[float] | Sequence[float]) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __truediv__(self, other: float | Series[int] | Series[float] | Sequence[float]) -> TimestampSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def unique(self) -> DatetimeArray: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def mean(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timestamp: ...
    def median(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timestamp: ...
    def std(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = ...,
        ddof: int = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def diff(self, periods: int = ...) -> TimedeltaSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def cumprod(
        self,
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Never: ...

class TimedeltaSeries(_SeriesSubclassBase[Timedelta, np.timedelta64]):
    # ignores needed because of mypy
    @overload  # type: ignore[override]
    def __add__(self, other: Period) -> PeriodSeries: ...
    @overload
    def __add__(
        self, other: datetime | Timestamp | TimestampSeries | DatetimeIndex
    ) -> TimestampSeries: ...
    @overload
    def __add__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: timedelta | Timedelta | np.timedelta64
    ) -> TimedeltaSeries: ...
    def __radd__(self, other: datetime | Timestamp | TimestampSeries) -> TimestampSeries: ...  # type: ignore[override]
    def __mul__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: num | Sequence[num] | Series[int] | Series[float]
    ) -> TimedeltaSeries: ...
    def unique(self) -> TimedeltaArray: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __sub__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta | Timedelta | TimedeltaSeries | TimedeltaIndex | np.timedelta64
        ),
    ) -> TimedeltaSeries: ...
    @overload  # type: ignore[override]
    def __truediv__(self, other: float | Sequence[float]) -> Self: ...
    @overload
    def __truediv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[float]: ...
    def __rtruediv__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[float]: ...
    @overload  # type: ignore[override]
    def __floordiv__(self, other: float | Sequence[float]) -> Self: ...
    @overload
    def __floordiv__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[int]: ...
    def __rfloordiv__(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        other: (
            timedelta
            | TimedeltaSeries
            | np.timedelta64
            | TimedeltaIndex
            | Sequence[timedelta]
        ),
    ) -> Series[int]: ...
    @property
    def dt(self) -> TimedeltaProperties: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def mean(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def median(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def std(  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool | None = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool = ...,
        **kwargs: Any,
    ) -> Timedelta: ...
    def diff(self, periods: int = ...) -> TimedeltaSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def cumsum(
        self,
        axis: AxisIndex | None = 0,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> TimedeltaSeries: ...
    def cumprod(
        self,
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Never: ...

class PeriodSeries(_SeriesSubclassBase[Period, np.object_]):
    @property
    def dt(self) -> PeriodProperties: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def __sub__(self, other: PeriodSeries) -> OffsetSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def diff(self, periods: int = ...) -> OffsetSeries: ...  # type: ignore[override] # pyright: ignore[reportIncompatibleMethodOverride]
    def cumprod(
        self,
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Never: ...

class OffsetSeries(_SeriesSubclassBase[BaseOffset, np.object_]):
    @overload  # type: ignore[override]
    def __radd__(self, other: Period) -> PeriodSeries: ...
    @overload
    def __radd__(self, other: BaseOffset) -> OffsetSeries: ...
    def cumprod(
        self,
        axis: AxisIndex = ...,
        skipna: _bool = ...,
        *args: Any,
        **kwargs: Any,
    ) -> Never: ...

class IntervalSeries(
    _SeriesSubclassBase[Interval[_OrderableT], np.object_], Generic[_OrderableT]
):
    @property
    def array(self) -> IntervalArray: ...
    def diff(self, periods: int = ...) -> Never: ...
