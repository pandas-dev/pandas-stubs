from datetime import (
    date,
    time,
)
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    Union,
    overload,
)

from matplotlib.axes import (
    Axes as PlotAxes,
    SubplotBase as SubplotBase,
)
import numpy as np
from pandas import (
    Period,
    Timedelta,
    Timestamp,
)
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.groupby.generic import (
    _SeriesGroupByNonScalar,
    _SeriesGroupByScalar,
)
from pandas.core.indexes.accessors import (
    CombinedDatetimelikeProperties,
    DatetimeProperties,
    TimedeltaProperties,
)
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.period import PeriodIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.indexing import (
    _AtIndexer,
    _iAtIndexer,
)
from pandas.core.resample import Resampler
from pandas.core.strings import StringMethods
from pandas.core.window import ExponentialMovingWindow
from pandas.core.window.rolling import (
    Rolling,
    Window,
)

from pandas._typing import (
    S1,
    ArrayLike,
    Axes,
    Axis,
    AxisType,
    Dtype,
    DtypeNp,
    FilePathOrBuffer,
    GroupByObjectNonScalar,
    HashableT,
    IgnoreRaise,
    IndexingInt,
    Label,
    Level,
    ListLike,
    MaskType,
    Renamer,
    Scalar,
    SeriesAxisType,
    np_ndarray_anyint,
    num,
)

from pandas.plotting import PlotAccessor

from .base import IndexOpsMixin
from .frame import DataFrame
from .generic import NDFrame
from .indexes.multi import MultiIndex
from .indexing import (
    _iLocIndexer,
    _LocIndexer,
)

_bool = bool
_str = str

class _iLocIndexerSeries(_iLocIndexer, Generic[S1]):
    # get item
    @overload
    def __getitem__(self, idx: IndexingInt) -> S1: ...
    @overload
    def __getitem__(self, idx: Index | slice | np_ndarray_anyint) -> Series[S1]: ...
    # set item
    @overload
    def __setitem__(self, idx: int, value: S1) -> None: ...
    @overload
    def __setitem__(
        self, idx: Index | slice | np_ndarray_anyint, value: S1 | Series[S1]
    ) -> None: ...

class _LocIndexerSeries(_LocIndexer, Generic[S1]):
    @overload
    def __getitem__(
        self,
        idx: MaskType
        | Index
        | Sequence[float]
        | list[_str]
        | slice
        | tuple[str | float | slice | Index, ...],
    ) -> Series[S1]: ...
    @overload
    def __getitem__(
        self,
        idx: _str | float,
    ) -> S1: ...
    @overload
    def __setitem__(
        self,
        idx: Index | MaskType,
        value: S1 | ArrayLike | Series[S1],
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: str,
        value: S1,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: list[int] | list[str] | list[str | int],
        value: S1 | ArrayLike | Series[S1],
    ) -> None: ...

class Series(IndexOpsMixin, NDFrame, Generic[S1]):

    _ListLike = Union[ArrayLike, dict[_str, np.ndarray], list, tuple, Index]
    __hash__: ClassVar[None]

    @overload
    def __new__(
        cls,
        data: DatetimeIndex,
        index: _str | int | Series | list | Index | None = ...,
        dtype=...,
        name: Hashable | None = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> TimestampSeries: ...
    @overload
    def __new__(
        cls,
        data: PeriodIndex,
        index: _str | int | Series | list | Index | None = ...,
        dtype=...,
        name: Hashable | None = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> Series[Period]: ...
    @overload
    def __new__(
        cls,
        data: object | _ListLike | Series[S1] | dict[int, S1] | dict[_str, S1] | None,
        dtype: type[S1],
        index: _str | int | Series | list | Index | None = ...,
        name: Hashable | None = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> Series[S1]: ...
    @overload
    def __new__(
        cls,
        data: object
        | _ListLike
        | Series[S1]
        | dict[int, S1]
        | dict[_str, S1]
        | None = ...,
        index: _str | int | Series | list | Index | None = ...,
        dtype=...,
        name: Hashable | None = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> Series: ...
    @property
    def hasnans(self) -> bool: ...
    def div(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[float]: ...
    def rdiv(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    @property
    def dtype(self) -> Dtype: ...
    @property
    def dtypes(self) -> Dtype: ...
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
    def __array_ufunc__(self, ufunc: Callable, method: _str, *inputs, **kwargs): ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    @property
    def axes(self) -> list: ...
    def take(
        self,
        indices: Sequence,
        axis: SeriesAxisType = ...,
        is_copy: _bool | None = ...,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def __getitem__(
        self,
        idx: list[_str]
        | Index
        | Series[S1]
        | slice
        | MaskType
        | tuple[S1 | slice, ...],
    ) -> Series: ...
    @overload
    def __getitem__(self, idx: int | _str) -> S1: ...
    def __setitem__(self, key, value) -> None: ...
    def repeat(
        self, repeats: int | list[int], axis: SeriesAxisType | None = ...
    ) -> Series[S1]: ...
    @property
    def index(self) -> Index | MultiIndex: ...
    @index.setter
    def index(self, idx: Index) -> None: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | None,
        drop: Literal[True],
        *,
        name: object | None = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Level | None,
        drop: Literal[True],
        *,
        name: object | None = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        /,
        drop: Literal[True],
        level: Sequence[Level] | None = ...,
        name: object | None = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        /,
        drop: Literal[True],
        level: Level | None = ...,
        name: object | None = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Sequence[Level] | None = ...,
        drop: Literal[False] = ...,
        name: object | None = ...,
        inplace: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Level | None = ...,
        drop: Literal[False] = ...,
        name: object | None = ...,
        inplace: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def to_string(
        self,
        buf: FilePathOrBuffer | None,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: _str | None = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        line_width: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: _str | None = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: _str | None = ...,
        max_rows: int | None = ...,
        min_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        line_width: int | None = ...,
        max_colwidth: int | None = ...,
        encoding: _str | None = ...,
    ) -> _str: ...
    @overload
    def to_markdown(
        self,
        buf: FilePathOrBuffer | None,
        mode: _str | None = ...,
        index: _bool = ...,
        storage_options: dict | None = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def to_markdown(
        self,
        mode: _str | None = ...,
        index: _bool = ...,
        storage_options: dict | None = ...,
    ) -> _str: ...
    def items(self) -> Iterable[tuple[Hashable, S1]]: ...
    def iteritems(self) -> Iterable[tuple[Label, S1]]: ...
    def keys(self) -> list: ...
    def to_dict(self, into: Hashable = ...) -> dict[Any, S1]: ...
    def to_frame(self, name: object | None = ...) -> DataFrame: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        axis: SeriesAxisType = ...,
        level: Level | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        squeeze: _bool = ...,
        observed: _bool = ...,
        dropna: _bool = ...,
    ) -> _SeriesGroupByScalar: ...
    @overload
    def groupby(
        self,
        by: GroupByObjectNonScalar = ...,
        axis: SeriesAxisType = ...,
        level: Level | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        squeeze: _bool = ...,
        observed: _bool = ...,
        dropna: _bool = ...,
    ) -> _SeriesGroupByNonScalar: ...
    @overload
    def count(self, level: None = ...) -> int: ...
    @overload
    def count(self, level: Hashable) -> Series[S1]: ...
    def mode(self, dropna=...) -> Series[S1]: ...
    def unique(self) -> np.ndarray: ...
    @overload
    def drop_duplicates(
        self, keep: Literal["first", "last", False] = ..., inplace: Literal[False] = ...
    ) -> Series[S1]: ...
    @overload
    def drop_duplicates(
        self, keep: Literal["first", "last", False], inplace: Literal[True]
    ) -> None: ...
    @overload
    def drop_duplicates(self, *, inplace: Literal[True]) -> None: ...
    @overload
    def drop_duplicates(
        self, keep: Literal["first", "last", False] = ..., inplace: bool = ...
    ) -> Series[S1] | None: ...
    def duplicated(
        self, keep: Literal["first", "last", False] = ...
    ) -> Series[_bool]: ...
    def idxmax(
        self, axis: SeriesAxisType = ..., skipna: _bool = ..., *args, **kwargs
    ) -> int | _str: ...
    def idxmin(
        self, axis: SeriesAxisType = ..., skipna: _bool = ..., *args, **kwargs
    ) -> int | _str: ...
    def round(self, decimals: int = ..., *args, **kwargs) -> Series[S1]: ...
    @overload
    def quantile(
        self,
        q: float = ...,
        interpolation: _str
        | Literal["linear", "lower", "higher", "midpoint", "nearest"] = ...,
    ) -> float: ...
    @overload
    def quantile(
        self,
        q: _ListLike,
        interpolation: _str
        | Literal["linear", "lower", "higher", "midpoint", "nearest"] = ...,
    ) -> Series[S1]: ...
    def corr(
        self,
        other: Series[S1],
        method: Literal["pearson", "kendall", "spearman"] = ...,
        min_periods: int = ...,
    ) -> float: ...
    def cov(
        self, other: Series[S1], min_periods: int | None = ..., ddof: int = ...
    ) -> float: ...
    def diff(self, periods: int = ...) -> Series[S1]: ...
    def autocorr(self, lag: int = ...) -> float: ...
    @overload
    def dot(self, other: Series[S1]) -> Scalar: ...
    @overload
    def dot(self, other: DataFrame) -> Series[S1]: ...
    @overload
    def dot(self, other: _ListLike) -> np.ndarray: ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...
    @overload
    def searchsorted(
        self,
        value: _ListLike,
        side: _str | Literal["left", "right"] = ...,
        sorter: _ListLike | None = ...,
    ) -> list[int]: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: _str | Literal["left", "right"] = ...,
        sorter: _ListLike | None = ...,
    ) -> int: ...
    def append(
        self,
        to_append: Series | Sequence[Series],
        ignore_index: _bool = ...,
        verify_integrity: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def compare(
        self,
        other: Series,
        align_axis: SeriesAxisType,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
    ) -> Series: ...
    @overload
    def compare(
        self,
        other: Series,
        align_axis: Literal["columns", 1] = ...,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
    ) -> DataFrame: ...
    def combine(
        self, other: Series[S1], func: Callable, fill_value: Scalar | None = ...
    ) -> Series[S1]: ...
    def combine_first(self, other: Series[S1]) -> Series[S1]: ...
    def update(self, other: Series[S1] | Sequence[S1] | Mapping[int, S1]) -> None: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[True],
        key: Callable | None = ...,
    ) -> None: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[False] = ...,
        key: Callable | None = ...,
    ) -> Series[S1]: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: _bool | Sequence[_bool] = ...,
        inplace: _bool | None = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        ignore_index: _bool = ...,
        key: Callable | None = ...,
    ) -> Series[S1] | None: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Level | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[True],
        key: Callable | None = ...,
    ) -> None: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Level | list[int] | list[_str] | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[False] = ...,
        key: Callable | None = ...,
    ) -> Series: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Level | list[int] | list[_str] | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        inplace: _bool | None = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        key: Callable | None = ...,
    ) -> Series | None: ...
    def argsort(
        self,
        axis: SeriesAxisType = ...,
        kind: _str | Literal["mergesort", "quicksort", "heapsort"] = ...,
        order: None = ...,
    ) -> Series[int]: ...
    def nlargest(
        self, n: int = ..., keep: _str | Literal["first", "last", "all"] = ...
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = ..., keep: _str | Literal["first", "last", "all"] = ...
    ) -> Series[S1]: ...
    def swaplevel(
        self, i: Level = ..., j: Level = ..., copy: _bool = ...
    ) -> Series[S1]: ...
    def reorder_levels(self, order: list) -> Series[S1]: ...
    def explode(self) -> Series[S1]: ...
    def unstack(
        self,
        level: Level = ...,
        fill_value: int | _str | dict | None = ...,
    ) -> DataFrame: ...
    def map(
        self, arg, na_action: _str | Literal["ignore"] | None = ...
    ) -> Series[S1]: ...
    def aggregate(
        self,
        func: Callable
        | _str
        | list[Callable | _str]
        | dict[SeriesAxisType, Callable | _str],
        axis: SeriesAxisType = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def agg(
        self,
        func: Callable
        | _str
        | list[Callable | _str]
        | dict[SeriesAxisType, Callable | _str] = ...,
        axis: SeriesAxisType = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def transform(
        self,
        func: list[Callable] | dict[_str, Callable],
        axis: SeriesAxisType = ...,
        *args,
        **kwargs,
    ) -> Series[S1]: ...
    def apply(
        self, func: Callable, convertDType: _bool = ..., args: tuple = ..., **kwds
    ) -> Series | DataFrame: ...
    def align(
        self,
        other: DataFrame | Series,
        join: _str | Literal["inner", "outer", "left", "right"] = ...,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        copy: _bool = ...,
        fill_value=...,
        method: _str | Literal["backfill", "bfill", "pad", "ffill"] | None = ...,
        limit: int | None = ...,
        fill_axis: SeriesAxisType = ...,
        broadcast_axis: SeriesAxisType | None = ...,
    ) -> tuple[Series, Series]: ...
    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
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
        index: Renamer | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @overload
    def rename(
        self,
        index: Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @overload
    def rename(
        self,
        index: Renamer | Hashable | None = ...,
        *,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None: ...
    def reindex_like(
        self,
        other: Series[S1],
        method: _str
        | Literal["backfill", "bfill", "pad", "ffill", "nearest"]
        | None = ...,
        copy: _bool = ...,
        limit: int | None = ...,
        tolerance: float | None = ...,
    ) -> Series: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[HashableT] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[HashableT] = ...,
        columns: Hashable | list[HashableT] = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[HashableT] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[HashableT] = ...,
        columns: Hashable | list[HashableT] = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[HashableT] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[HashableT] = ...,
        columns: Hashable | list[HashableT] = ...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Series | None: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series[S1] | DataFrame | None = ...,
        method: _str | Literal["backfill", "bfill", "pad", "ffill"] | None = ...,
        axis: SeriesAxisType = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series[S1] | DataFrame | None = ...,
        method: _str | Literal["backfill", "bfill", "pad", "ffill"] | None = ...,
        axis: SeriesAxisType = ...,
        *,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series[S1]: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series[S1] | DataFrame | None = ...,
        method: _str | Literal["backfill", "bfill", "pad", "ffill"] | None = ...,
        axis: SeriesAxisType = ...,
        inplace: _bool = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series[S1] | None: ...
    @overload
    def replace(
        self,
        to_replace: _str | list | dict | Series[S1] | float | None = ...,
        value: Scalar | dict | list | _str | None = ...,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        regex=...,
        method: _str | Literal["pad", "ffill", "bfill"] | None = ...,
    ) -> Series[S1]: ...
    @overload
    def replace(
        self,
        to_replace: _str | list | dict | Series[S1] | float | None = ...,
        value: Scalar | dict | list | _str | None = ...,
        limit: int | None = ...,
        regex=...,
        method: _str | Literal["pad", "ffill", "bfill"] | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace: _str | list | dict | Series[S1] | float | None = ...,
        value: Scalar | dict | list | _str | None = ...,
        inplace: _bool = ...,
        limit: int | None = ...,
        regex=...,
        method: _str | Literal["pad", "ffill", "bfill"] | None = ...,
    ) -> Series[S1] | None: ...
    def shift(
        self,
        periods: int = ...,
        freq=...,
        axis: SeriesAxisType = ...,
        fill_value: object | None = ...,
    ) -> Series[S1]: ...
    def memory_usage(self, index: _bool = ..., deep: _bool = ...) -> int: ...
    def isin(self, values: Iterable | Series[S1] | dict) -> Series[_bool]: ...
    def between(
        self,
        left: Scalar | Sequence,
        right: Scalar | Sequence,
        inclusive: Literal["both", "neither", "left", "right"] = ...,
    ) -> Series[_bool]: ...
    def isna(self) -> Series[_bool]: ...
    def isnull(self) -> Series[_bool]: ...
    def notna(self) -> Series[_bool]: ...
    def notnull(self) -> Series[_bool]: ...
    @overload
    def dropna(
        self,
        axis: SeriesAxisType = ...,
        how: _str | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def dropna(
        self,
        axis: SeriesAxisType = ...,
        inplace: _bool = ...,
        how: _str | None = ...,
    ) -> Series[S1]: ...
    def to_timestamp(
        self,
        freq=...,
        how: _str | Literal["start", "end", "s", "e"] = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def to_period(self, freq: _str | None = ..., copy: _bool = ...) -> DataFrame: ...
    @property
    def str(self) -> StringMethods[Series]: ...
    @property
    def dt(self) -> CombinedDatetimelikeProperties: ...
    @property
    def plot(self) -> PlotAccessor: ...
    sparse = ...
    def hist(
        self,
        by: object | None = ...,
        ax: PlotAxes | None = ...,
        grid: _bool = ...,
        xlabelsize: int | None = ...,
        xrot: float | None = ...,
        ylabelsize: int | None = ...,
        yrot: float | None = ...,
        figsize: tuple[float, float] | None = ...,
        bins: int | Sequence = ...,
        backend: _str | None = ...,
        **kwargs,
    ) -> SubplotBase: ...
    def swapaxes(
        self, axis1: SeriesAxisType, axis2: SeriesAxisType, copy: _bool = ...
    ) -> Series[S1]: ...
    def droplevel(
        self, level: Level | list[Level], axis: SeriesAxisType = ...
    ) -> DataFrame: ...
    def pop(self, item: _str) -> Series[S1]: ...
    def squeeze(self, axis: SeriesAxisType | None = ...) -> Scalar: ...
    def __abs__(self) -> Series[S1]: ...
    def add_prefix(self, prefix: _str) -> Series[S1]: ...
    def add_suffix(self, suffix: _str) -> Series[S1]: ...
    def reindex(
        self,
        index: Axes | None = ...,
        method: Literal["backfill", "bfill", "pad", "ffill", "nearest"] | None = ...,
        copy: bool = ...,
        level: int | _str = ...,
        fill_value: Scalar | None = ...,
        limit: int | None = ...,
        tolerance: float | None = ...,
    ) -> Series[S1]: ...
    def filter(
        self,
        items: _ListLike | None = ...,
        like: _str | None = ...,
        regex: _str | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def head(self, n: int = ...) -> Series[S1]: ...
    def tail(self, n: int = ...) -> Series[S1]: ...
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: _bool = ...,
        weights: _str | _ListLike | np.ndarray | None = ...,
        random_state: int | None = ...,
        axis: SeriesAxisType | None = ...,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    def astype(
        self,
        dtype: S1 | _str | type[Scalar],
        copy: _bool = ...,
        errors: _str | Literal["raise", "ignore"] = ...,
    ) -> Series: ...
    def copy(self, deep: _bool = ...) -> Series[S1]: ...
    def infer_objects(self) -> Series[S1]: ...
    def convert_dtypes(
        self,
        infer_objects: _bool = ...,
        convert_string: _bool = ...,
        convert_integer: _bool = ...,
        convert_boolean: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def ffill(
        self,
        axis: SeriesAxisType | None = ...,
        *,
        inplace: Literal[True],
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        axis: SeriesAxisType | None = ...,
        *,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        axis: SeriesAxisType | None = ...,
        *,
        inplace: Literal[True],
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        axis: SeriesAxisType | None = ...,
        *,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        value: S1 | dict | Series[S1] | DataFrame,
        axis: SeriesAxisType = ...,
        inplace: _bool = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> Series[S1] | None: ...
    def interpolate(
        self,
        method: _str
        | Literal[
            "linear",
            "time",
            "index",
            "values",
            "pad",
            "nearest",
            "slinear",
            "quadratic",
            "cubic",
            "spline",
            "barycentric",
            "polynomial",
            "krogh",
            "pecewise_polynomial",
            "spline",
            "pchip",
            "akima",
            "from_derivatives",
        ] = ...,
        axis: SeriesAxisType | None = ...,
        limit: int | None = ...,
        inplace: _bool = ...,
        limit_direction: _str | Literal["forward", "backward", "both"] | None = ...,
        limit_area: _str | Literal["inside", "outside"] | None = ...,
        downcast: _str | Literal["infer"] | None = ...,
        **kwargs,
    ) -> Series[S1]: ...
    def asof(
        self,
        where: Scalar | Sequence[Scalar],
        subset: _str | Sequence[_str] | None = ...,
    ) -> Scalar | Series[S1]: ...
    def clip(
        self,
        lower: float | None = ...,
        upper: float | None = ...,
        axis: SeriesAxisType | None = ...,
        inplace: _bool = ...,
        *args,
        **kwargs,
    ) -> Series[S1]: ...
    def asfreq(
        self,
        freq,
        method: _str | Literal["backfill", "bfill", "pad", "ffill"] | None = ...,
        how: _str | Literal["start", "end"] | None = ...,
        normalize: _bool = ...,
        fill_value: Scalar | None = ...,
    ) -> Series[S1]: ...
    def at_time(
        self,
        time: _str | time,
        asof: _bool = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def between_time(
        self,
        start_time: _str | time,
        end_time: _str | time,
        include_start: _bool = ...,
        include_end: _bool = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def resample(
        self,
        rule,
        axis: SeriesAxisType = ...,
        closed: _str | None = ...,
        label: _str | None = ...,
        convention: _str | Literal["start", "end", "s", "e"] = ...,
        kind: _str | Literal["timestamp", "period"] | None = ...,
        loffset=...,
        base: int = ...,
        on: _str | None = ...,
        level: Level | None = ...,
        origin: Timestamp
        | Literal["epoch", "start", "start_day", "end", "end_day"] = ...,
        offset: Timedelta | _str | None = ...,
    ) -> Resampler: ...
    def first(self, offset) -> Series[S1]: ...
    def last(self, offset) -> Series[S1]: ...
    def rank(
        self,
        axis: SeriesAxisType = ...,
        method: _str | Literal["average", "min", "max", "first", "dense"] = ...,
        numeric_only: _bool | None = ...,
        na_option: _str | Literal["keep", "top", "bottom"] = ...,
        ascending: _bool = ...,
        pct: _bool = ...,
    ) -> Series: ...
    def where(
        self,
        cond: Series[S1] | Series[_bool] | np.ndarray,
        other=...,
        inplace: _bool = ...,
        axis: SeriesAxisType | None = ...,
        level: Level | None = ...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ) -> Series[S1]: ...
    def mask(
        self,
        cond: MaskType,
        other: Scalar | Series[S1] | DataFrame | Callable = ...,
        inplace: _bool = ...,
        axis: SeriesAxisType | None = ...,
        level: Level | None = ...,
        errors: _str | Literal["raise", "ignore"] = ...,
        try_cast: _bool = ...,
    ) -> Series[S1]: ...
    def slice_shift(
        self, periods: int = ..., axis: SeriesAxisType = ...
    ) -> Series[S1]: ...
    def tshift(
        self, periods: int = ..., freq=..., axis: SeriesAxisType = ...
    ) -> Series[S1]: ...
    def truncate(
        self,
        before: date | _str | int | None = ...,
        after: date | _str | int | None = ...,
        axis: SeriesAxisType | None = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def tz_convert(
        self,
        tz,
        axis: SeriesAxisType = ...,
        level: Level | None = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def tz_localize(
        self,
        tz,
        axis: SeriesAxisType = ...,
        level: Level | None = ...,
        copy: _bool = ...,
        ambiguous=...,
        nonexistent: _str = ...,
    ) -> Series[S1]: ...
    def abs(self) -> Series[S1]: ...
    def describe(
        self,
        percentiles: list[float] | None = ...,
        include: _str | Literal["all"] | list[S1] | None = ...,
        exclude: S1 | list[S1] | None = ...,
        datetime_is_numeric: _bool | None = ...,
    ) -> Series[S1]: ...
    def pct_change(
        self,
        periods: int = ...,
        fill_method: _str = ...,
        limit: int | None = ...,
        freq=...,
        **kwargs,
    ) -> Series[S1]: ...
    def first_valid_index(self) -> Scalar: ...
    def last_valid_index(self) -> Scalar: ...
    def value_counts(
        self,
        normalize: _bool = ...,
        sort: _bool = ...,
        ascending: _bool = ...,
        bins: int | None = ...,
        dropna: _bool = ...,
    ) -> Series[S1]: ...
    def transpose(self, *args, **kwargs) -> Series[S1]: ...
    @property
    def T(self) -> Series[S1]: ...
    # The rest of these were left over from the old
    # stubs we shipped in preview. They may belong in
    # the base classes in some cases; I expect stubgen
    # just failed to generate these so I couldn't match
    # them up.
    @overload
    def __add__(self, other: TimestampSeries) -> TimestampSeries: ...
    @overload
    def __add__(self, other: DatetimeIndex) -> TimestampSeries: ...
    @overload
    def __add__(self, other: Timestamp) -> TimestampSeries: ...
    @overload
    def __add__(
        self, other: num | _str | Timedelta | _ListLike | Series[S1]
    ) -> Series: ...
    def __and__(self, other: _ListLike | Series[S1]) -> Series[_bool]: ...
    # def __array__(self, dtype: Optional[_bool] = ...) -> _np_ndarray
    def __div__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __eq__(self, other: object) -> Series[_bool]: ...  # type: ignore[override]
    def __floordiv__(self, other: num | _ListLike | Series[S1]) -> Series[int]: ...
    def __ge__(self, other: S1 | _ListLike | Series[S1]) -> Series[_bool]: ...
    def __gt__(self, other: S1 | _ListLike | Series[S1]) -> Series[_bool]: ...
    # def __iadd__(self, other: S1) -> Series[S1]: ...
    # def __iand__(self, other: S1) -> Series[_bool]: ...
    # def __idiv__(self, other: S1) -> Series[S1]: ...
    # def __ifloordiv__(self, other: S1) -> Series[S1]: ...
    # def __imod__(self, other: S1) -> Series[S1]: ...
    # def __imul__(self, other: S1) -> Series[S1]: ...
    # def __ior__(self, other: S1) -> Series[_bool]: ...
    # def __ipow__(self, other: S1) -> Series[S1]: ...
    # def __isub__(self, other: S1) -> Series[S1]: ...
    # def __itruediv__(self, other: S1) -> Series[S1]: ...
    # def __itruediv__(self, other) -> None: ...
    # def __ixor__(self, other: S1) -> Series[_bool]: ...
    def __le__(self, other: S1 | _ListLike | Series[S1]) -> Series[_bool]: ...
    def __lt__(self, other: S1 | _ListLike | Series[S1]) -> Series[_bool]: ...
    @overload
    def __mul__(self, other: Timedelta | TimedeltaSeries) -> TimedeltaSeries: ...
    @overload
    def __mul__(self, other: num | _ListLike | Series) -> Series: ...
    def __mod__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __ne__(self, other: object) -> Series[_bool]: ...  # type: ignore[override]
    def __pow__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __or__(self, other: _ListLike | Series[S1]) -> Series[_bool]: ...
    def __radd__(self, other: num | _str | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rand__(self, other: num | _ListLike | Series[S1]) -> Series[_bool]: ...
    def __rdiv__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rdivmod__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rfloordiv__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rmod__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rmul__(self, other: num | _ListLike | Series) -> Series: ...
    def __rnatmul__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __rpow__(self, other: num | _ListLike | Series[S1]) -> Series[S1]: ...
    def __ror__(self, other: num | _ListLike | Series[S1]) -> Series[_bool]: ...
    def __rsub__(self, other: num | _ListLike | Series[S1]) -> Series: ...
    @overload
    def __rtruediv__(self, other: Timedelta | TimedeltaSeries) -> Series[float]: ...
    @overload
    def __rtruediv__(self, other: num | _ListLike | Series[S1]) -> Series: ...
    def __rxor__(self, other: num | _ListLike | Series[S1]) -> Series[_bool]: ...
    @overload
    def __sub__(self, other: Timestamp | TimestampSeries) -> TimedeltaSeries: ...
    @overload
    def __sub__(
        self, other: Timedelta | TimedeltaSeries | TimedeltaIndex
    ) -> TimestampSeries: ...
    @overload
    def __sub__(self, other: num | _ListLike | Series) -> Series: ...
    @overload
    def __truediv__(
        self, other: Timedelta | TimedeltaSeries | TimedeltaIndex
    ) -> Series[float]: ...
    @overload
    def __truediv__(self, other: num | _ListLike | Series[S1]) -> Series: ...
    def __xor__(self, other: _ListLike | Series[S1]) -> Series: ...
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
    # Methods
    def add(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: int = ...,
    ) -> Series[S1]: ...
    def all(
        self,
        axis: SeriesAxisType = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        level: Level | None = ...,
        **kwargs,
    ) -> _bool: ...
    def any(
        self,
        axis: SeriesAxisType = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        level: Level | None = ...,
        **kwargs,
    ) -> _bool: ...
    def cummax(
        self, axis: SeriesAxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def cummin(
        self, axis: SeriesAxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def cumprod(
        self, axis: SeriesAxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def cumsum(
        self, axis: SeriesAxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def divide(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[float]: ...
    def divmod(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def eq(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def ewm(
        self,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | None = ...,
        alpha: float | None = ...,
        min_periods: int = ...,
        adjust: _bool = ...,
        ignore_na: _bool = ...,
        axis: SeriesAxisType = ...,
    ) -> ExponentialMovingWindow: ...
    def expanding(
        self, min_periods: int = ..., center: _bool = ..., axis: SeriesAxisType = ...
    ) -> DataFrame: ...
    def floordiv(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[int]: ...
    def ge(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def gt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def item(self) -> S1: ...
    @overload
    def kurt(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def kurt(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def kurtosis(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level | None,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def kurtosis(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Scalar: ...
    def le(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def lt(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    @overload
    def mad(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def mad(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def max(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def max(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        *,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> S1: ...
    @overload
    def mean(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def mean(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> float: ...
    @overload
    def median(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def median(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> float: ...
    @overload
    def min(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def min(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> S1: ...
    def mod(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def mul(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def multiply(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def ne(
        self,
        other: Scalar | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def nunique(self, dropna: _bool = ...) -> int: ...
    def pow(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    @overload
    def prod(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def prod(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def product(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def product(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Scalar: ...
    def radd(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rdivmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rfloordiv(
        self,
        other,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rmod(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rmul(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    @overload
    def rolling(
        self,
        window,
        min_periods: int | None = ...,
        center: _bool = ...,
        *,
        win_type: _str,
        on: _str | None = ...,
        axis: SeriesAxisType = ...,
        closed: _str | None = ...,
    ) -> Window: ...
    @overload
    def rolling(
        self,
        window,
        min_periods: int | None = ...,
        center: _bool = ...,
        *,
        on: _str | None = ...,
        axis: SeriesAxisType = ...,
        closed: _str | None = ...,
    ) -> Rolling: ...
    def rpow(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rsub(
        self,
        other: Series[S1] | Scalar,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rtruediv(
        self,
        other,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    @overload
    def sem(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def sem(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def skew(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def skew(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def std(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[float]: ...
    @overload
    def std(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> float: ...
    def sub(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def subtract(
        self,
        other: num | _ListLike | Series[S1],
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType | None = ...,
    ) -> Series[S1]: ...
    def sum(
        self: Series[S1],
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: Level | None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        **kwargs,
    ) -> S1: ...
    def to_list(self) -> list[S1]: ...
    def to_numpy(
        self,
        dtype: type[DtypeNp] | None = ...,
        copy: _bool = ...,
        na_value=...,
        **kwargs,
    ) -> np.ndarray: ...
    def to_records(
        self,
        index: _bool = ...,
        columnDTypes: _str | dict | None = ...,
        indexDTypes: _str | dict | None = ...,
    ): ...
    def tolist(self) -> list[S1]: ...
    def truediv(
        self,
        other,
        level: Level | None = ...,
        fill_value: float | None = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[float]: ...
    @overload
    def var(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def var(
        self,
        axis: SeriesAxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike = ...,
        index: Scalar | ListLike | Callable | dict | None = ...,
        columns: Scalar | ListLike | Callable | dict | None = ...,
        axis: SeriesAxisType | None = ...,
        copy: _bool = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def rename_axis(
        self,
        mapper: Scalar | ListLike = ...,
        index: Scalar | ListLike | Callable | dict | None = ...,
        columns: Scalar | ListLike | Callable | dict | None = ...,
        axis: SeriesAxisType | None = ...,
        copy: _bool = ...,
        inplace: Literal[False] = ...,
    ) -> Series: ...
    @overload
    def set_axis(
        self, labels, axis: Axis = ..., inplace: Literal[False] = ...
    ) -> Series[S1]: ...
    @overload
    def set_axis(self, labels, axis: Axis, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(self, labels, *, inplace: Literal[True]) -> None: ...
    @overload
    def set_axis(
        self, labels, axis: Axis = ..., inplace: bool = ...
    ) -> Series[S1] | None: ...
    def __iter__(self) -> Iterator[S1]: ...

class TimestampSeries(Series[Timestamp]):
    # ignore needed because of mypy
    @property
    def dt(self) -> DatetimeProperties: ...  # type: ignore[override]

class TimedeltaSeries(Series[Timedelta]):
    # ignore needed because of mypy
    def __mul__(self, other: num) -> TimedeltaSeries: ...  # type: ignore[override]
    def __sub__(  # type: ignore[override]
        self, other: Timedelta | TimedeltaSeries | TimedeltaIndex
    ) -> TimedeltaSeries: ...
    # ignore needed because of mypy
    @property
    def dt(self) -> TimedeltaProperties: ...  # type: ignore[override]
