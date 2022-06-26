import numpy as np
from datetime import time, date

from matplotlib.axes import Axes as PlotAxes, SubplotBase as SubplotBase
from .base import IndexOpsMixin
from .generic import NDFrame
from .indexes.multi import MultiIndex
from .indexing import _iLocIndexer, _LocIndexer
from .frame import DataFrame
from pandas.core.arrays.base import ExtensionArray
from pandas.core.groupby.generic import SeriesGroupBy
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
from pandas.core.resample import Resampler
from pandas.core.strings import StringMethods
from pandas.core.window.rolling import Rolling, Window
from pandas.core.window import ExponentialMovingWindow
from pandas._typing import (
    ArrayLike as ArrayLike,
    Axis as Axis,
    AxisType as AxisType,
    Dtype as Dtype,
    DtypeNp as DtypeNp,
    FilePathOrBuffer as FilePathOrBuffer,
    IgnoreRaise as IgnoreRaise,
    Level as Level,
    ListLike as ListLike,
    MaskType as MaskType,
    Renamer as Renamer,
    S1 as S1,
    Scalar as Scalar,
    SeriesAxisType as SeriesAxisType,
    Timestamp as Timestamp,
    Timedelta as Timedelta,
    num as num,
    Label as Label,
)
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
)

_bool = bool
_str = str

class _iLocIndexerSeries(_iLocIndexer, Generic[S1]):
    # get item
    @overload
    def __getitem__(self, idx: int) -> S1: ...
    @overload
    def __getitem__(self, idx: Union[Index, slice]) -> Series[S1]: ...
    # set item
    @overload
    def __setitem__(self, idx: int, value: S1) -> None: ...
    @overload
    def __setitem__(self, idx: Index, value: Union[S1, Series[S1]]) -> None: ...

class _LocIndexerSeries(_LocIndexer, Generic[S1]):
    @overload
    def __getitem__(
        self,
        idx: Union[
            MaskType,
            Index,
            Sequence[Union[int, float]],
            List[_str],
            slice,
            Tuple[Union[int, str, float, slice, Index], ...],
        ],
    ) -> Series[S1]: ...
    @overload
    def __getitem__(
        self,
        idx: Union[int, _str, float],
    ) -> S1: ...
    @overload
    def __setitem__(
        self,
        idx: Union[Index, MaskType],
        value: Union[S1, ArrayLike, Series[S1]],
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
        idx: Union[List[int], List[str], List[Union[str, int]]],
        value: Union[S1, ArrayLike, Series[S1]],
    ) -> None: ...

class Series(IndexOpsMixin, NDFrame, Generic[S1]):

    _ListLike = Union[ArrayLike, Dict[_str, np.ndarray], List, Tuple, Index]
    @overload
    def __new__(
        cls,
        data: DatetimeIndex,
        index: Union[_str, int, Series, List, Index] = ...,
        dtype=...,
        name: Optional[Hashable] = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> TimestampSeries: ...
    @overload
    def __new__(
        cls,
        data: Optional[
            Union[object, _ListLike, Series[S1], Dict[int, S1], Dict[_str, S1]]
        ],
        dtype: Type[S1],
        index: Union[_str, int, Series, List, Index] = ...,
        name: Optional[Hashable] = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> Series[S1]: ...
    @overload
    def __new__(
        cls,
        data: Optional[
            Union[object, _ListLike, Series[S1], Dict[int, S1], Dict[_str, S1]]
        ] = ...,
        index: Union[_str, int, Series, List, Index] = ...,
        dtype=...,
        name: Optional[Hashable] = ...,
        copy: bool = ...,
        fastpath: bool = ...,
    ) -> Series: ...
    @property
    def hasnans(self) -> bool: ...
    def div(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[float]: ...
    def rdiv(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    @property
    def dtype(self) -> Dtype: ...
    @property
    def dtypes(self) -> Dtype: ...
    @property
    def name(self) -> Optional[Hashable]: ...
    @name.setter
    def name(self, value: Optional[Hashable]) -> None: ...
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
    def axes(self) -> List: ...
    def take(
        self,
        indices: Sequence,
        axis: SeriesAxisType = ...,
        is_copy: Optional[_bool] = ...,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def __getitem__(
        self,
        idx: Union[
            List[_str], Index, Series[S1], slice, MaskType, Tuple[Union[S1, slice], ...]
        ],
    ) -> Series: ...
    @overload
    def __getitem__(self, idx: Union[int, _str]) -> S1: ...
    def __setitem__(self, key, value) -> None: ...
    def repeat(
        self, repeats: Union[int, List[int]], axis: Optional[SeriesAxisType] = ...
    ) -> Series[S1]: ...
    @property
    def index(self) -> Union[Index, MultiIndex]: ...
    @index.setter
    def index(self, idx: Index) -> None: ...
    @overload
    def reset_index(
        self,
        level: Optional[Sequence[Level]],
        drop: Literal[True],
        *,
        name: Optional[object] = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Optional[Level],
        drop: Literal[True],
        *,
        name: Optional[object] = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        /,
        drop: Literal[True],
        level: Optional[Sequence[Level]] = ...,
        name: Optional[object] = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        /,
        drop: Literal[True],
        level: Optional[Level] = ...,
        name: Optional[object] = ...,
        inplace: _bool = ...,
    ) -> Series[S1]: ...
    @overload
    def reset_index(
        self,
        level: Optional[Sequence[Level]] = ...,
        drop: Literal[False] = ...,
        name: Optional[object] = ...,
        inplace: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Optional[Level] = ...,
        drop: Literal[False] = ...,
        name: Optional[object] = ...,
        inplace: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def to_string(
        self,
        buf: Optional[FilePathOrBuffer],
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: Optional[_bool] = ...,
        index_names: _bool = ...,
        justify: Optional[_str] = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...,
        max_cols: Optional[int] = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        line_width: Optional[int] = ...,
        max_colwidth: Optional[int] = ...,
        encoding: Optional[_str] = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: Optional[_bool] = ...,
        index_names: _bool = ...,
        justify: Optional[_str] = ...,
        max_rows: Optional[int] = ...,
        min_rows: Optional[int] = ...,
        max_cols: Optional[int] = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        line_width: Optional[int] = ...,
        max_colwidth: Optional[int] = ...,
        encoding: Optional[_str] = ...,
    ) -> _str: ...
    @overload
    def to_markdown(
        self,
        buf: Optional[FilePathOrBuffer],
        mode: Optional[_str] = ...,
        index: _bool = ...,
        storage_options: Optional[dict] = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def to_markdown(
        self,
        mode: Optional[_str] = ...,
        index: _bool = ...,
        storage_options: Optional[dict] = ...,
    ) -> _str: ...
    def items(self) -> Iterable[Tuple[Hashable, S1]]: ...
    def iteritems(self) -> Iterable[Tuple[Label, S1]]: ...
    def keys(self) -> List: ...
    def to_dict(self, into: Hashable = ...) -> Dict[Any, Any]: ...
    def to_frame(self, name: Optional[object] = ...) -> DataFrame: ...
    def groupby(
        self,
        by=...,
        axis: SeriesAxisType = ...,
        level: Optional[Level] = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        squeeze: _bool = ...,
        observed: _bool = ...,
        dropna: _bool = ...,
    ) -> SeriesGroupBy: ...
    @overload
    def count(self, level: None = ...) -> int: ...
    @overload
    def count(self, level: Hashable) -> Series[S1]: ...
    def mode(self, dropna) -> Series[S1]: ...
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
    ) -> Optional[Series[S1]]: ...
    def duplicated(
        self, keep: Literal["first", "last", False] = ...
    ) -> Series[_bool]: ...
    def idxmax(
        self, axis: SeriesAxisType = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Union[int, _str]: ...
    def idxmin(
        self, axis: SeriesAxisType = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Union[int, _str]: ...
    def round(self, decimals: int = ..., *args, **kwargs) -> Series[S1]: ...
    @overload
    def quantile(
        self,
        q: float = ...,
        interpolation: Union[
            _str, Literal["linear", "lower", "higher", "midpoint", "nearest"]
        ] = ...,
    ) -> float: ...
    @overload
    def quantile(
        self,
        q: _ListLike,
        interpolation: Union[
            _str, Literal["linear", "lower", "higher", "midpoint", "nearest"]
        ] = ...,
    ) -> Series[S1]: ...
    def corr(
        self,
        other: Series[S1],
        method: Literal["pearson", "kendall", "spearman"] = ...,
        min_periods: int = ...,
    ) -> float: ...
    def cov(
        self, other: Series[S1], min_periods: Optional[int] = ..., ddof: int = ...
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
        side: Union[_str, Literal["left", "right"]] = ...,
        sorter: Optional[_ListLike] = ...,
    ) -> List[int]: ...
    @overload
    def searchsorted(
        self,
        value: Scalar,
        side: Union[_str, Literal["left", "right"]] = ...,
        sorter: Optional[_ListLike] = ...,
    ) -> int: ...
    def append(
        self,
        to_append: Union[Series, Sequence[Series]],
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
        self, other: Series[S1], func: Callable, fill_value: Optional[Scalar] = ...
    ) -> Series[S1]: ...
    def combine_first(self, other: Series[S1]) -> Series[S1]: ...
    def update(
        self, other: Union[Series[S1], Sequence[S1], Mapping[int, S1]]
    ) -> None: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[True],
        key: Optional[Callable] = ...,
    ) -> None: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[False],
        key: Optional[Callable] = ...,
    ) -> Series[S1]: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        *,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        key: Optional[Callable] = ...,
    ) -> Series[S1]: ...
    @overload
    def sort_values(
        self,
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        inplace: Optional[_bool] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        key: Optional[Callable] = ...,
    ) -> Union[None, Series[S1]]: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[True],
        key: Optional[Callable] = ...,
    ) -> None: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Optional[Union[Level, List[int], List[_str]]] = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[False],
        key: Optional[Callable] = ...,
    ) -> Series: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Optional[Union[Level, List[int], List[_str]]] = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        *,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        key: Optional[Callable] = ...,
    ) -> Series: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Optional[Union[Level, List[int], List[_str]]] = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        inplace: Optional[_bool] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        key: Optional[Callable] = ...,
    ) -> Union[None, Series]: ...
    def argsort(
        self,
        axis: SeriesAxisType = ...,
        kind: Union[_str, Literal["mergesort", "quicksort", "heapsort"]] = ...,
        order: None = ...,
    ) -> Series[int]: ...
    def nlargest(
        self, n: int = ..., keep: Union[_str, Literal["first", "last", "all"]] = ...
    ) -> Series[S1]: ...
    def nsmallest(
        self, n: int = ..., keep: Union[_str, Literal["first", "last", "all"]] = ...
    ) -> Series[S1]: ...
    def swaplevel(
        self, i: Level = ..., j: Level = ..., copy: _bool = ...
    ) -> Series[S1]: ...
    def reorder_levels(self, order: List) -> Series[S1]: ...
    def explode(self) -> Series[S1]: ...
    def unstack(
        self,
        level: Level = ...,
        fill_value: Optional[Union[int, _str, Dict]] = ...,
    ) -> DataFrame: ...
    def map(
        self, arg, na_action: Optional[Union[_str, Literal["ignore"]]] = ...
    ) -> Series[S1]: ...
    def aggregate(
        self,
        func: Union[
            Callable,
            _str,
            List[Union[Callable, _str]],
            Dict[SeriesAxisType, Union[Callable, _str]],
        ],
        axis: SeriesAxisType = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def agg(
        self,
        func: Union[
            Callable,
            _str,
            List[Union[Callable, _str]],
            Dict[SeriesAxisType, Union[Callable, _str]],
        ] = ...,
        axis: SeriesAxisType = ...,
        *args,
        **kwargs,
    ) -> None: ...
    def transform(
        self,
        func: Union[List[Callable], Dict[_str, Callable]],
        axis: SeriesAxisType = ...,
        *args,
        **kwargs,
    ) -> Series[S1]: ...
    def apply(
        self, func: Callable, convertDType: _bool = ..., args: Tuple = ..., **kwds
    ) -> Union[Series, DataFrame]: ...
    def align(
        self,
        other: Union[DataFrame, Series],
        join: Union[_str, Literal["inner", "outer", "left", "right"]] = ...,
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        copy: _bool = ...,
        fill_value=...,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill"]]
        ] = ...,
        limit: Optional[int] = ...,
        fill_axis: SeriesAxisType = ...,
        broadcast_axis: Optional[SeriesAxisType] = ...,
    ) -> Tuple[Series, Series]: ...
    @overload
    def rename(
        self,
        index: Optional[Union[Renamer, Hashable]] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        index: Optional[Renamer] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @overload
    def rename(
        self,
        index: Optional[Hashable] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @overload
    def rename(
        self,
        index: Optional[Union[Renamer, Hashable]] = ...,
        *,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> Optional[Series]: ...
    def reindex_like(
        self,
        other: Series[S1],
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill", "nearest"]]
        ] = ...,
        copy: _bool = ...,
        limit: Optional[int] = ...,
        tolerance: Optional[float] = ...,
    ) -> Series: ...
    @overload
    def drop(
        self,
        labels: Union[Hashable, List[Hashable]] = ...,
        *,
        axis: Axis = ...,
        index: Union[Hashable, List[Hashable]] = ...,
        columns: Union[Hashable, List[Hashable]] = ...,
        level: Optional[Level] = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: Union[Hashable, List[Hashable]] = ...,
        *,
        axis: Axis = ...,
        index: Union[Hashable, List[Hashable]] = ...,
        columns: Union[Hashable, List[Hashable]] = ...,
        level: Optional[Level] = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> Series: ...
    @overload
    def drop(
        self,
        labels: Union[Hashable, List[Hashable]] = ...,
        *,
        axis: Axis = ...,
        index: Union[Hashable, List[Hashable]] = ...,
        columns: Union[Hashable, List[Hashable]] = ...,
        level: Optional[Level] = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> Optional[Series]: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series[S1], DataFrame]] = ...,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill"]]
        ] = ...,
        axis: SeriesAxisType = ...,
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series[S1], DataFrame]] = ...,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill"]]
        ] = ...,
        axis: SeriesAxisType = ...,
        *,
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> Series[S1]: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series[S1], DataFrame]] = ...,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill"]]
        ] = ...,
        axis: SeriesAxisType = ...,
        inplace: _bool = ...,
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> Union[Series[S1], None]: ...
    def replace(
        self,
        to_replace: Optional[Union[_str, List, Dict, Series[S1], int, float]] = ...,
        value: Optional[Union[Scalar, Dict, List, _str]] = ...,
        inplace: _bool = ...,
        limit: Optional[int] = ...,
        regex=...,
        method: Optional[Union[_str, Literal["pad", "ffill", "bfill"]]] = ...,
    ) -> Series[S1]: ...
    def shift(
        self,
        periods: int = ...,
        freq=...,
        axis: SeriesAxisType = ...,
        fill_value: Optional[object] = ...,
    ) -> Series[S1]: ...
    def memory_usage(self, index: _bool = ..., deep: _bool = ...) -> int: ...
    def isin(self, values: Union[Iterable, Series[S1], Dict]) -> Series[_bool]: ...
    def between(
        self,
        left: Union[Scalar, Sequence],
        right: Union[Scalar, Sequence],
        inclusive: Literal["both", "neither", "left", "right"] = "both",
    ) -> Series[_bool]: ...
    def isna(self) -> Series[_bool]: ...
    def isnull(self) -> Series[_bool]: ...
    def notna(self) -> Series[_bool]: ...
    def notnull(self) -> Series[_bool]: ...
    @overload
    def dropna(
        self,
        axis: SeriesAxisType = ...,
        how: Optional[_str] = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def dropna(
        self,
        axis: SeriesAxisType = ...,
        inplace: _bool = ...,
        how: Optional[_str] = ...,
    ) -> Series[S1]: ...
    def to_timestamp(
        self,
        freq=...,
        how: Union[_str, Literal["start", "end", "s", "e"]] = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def to_period(self, freq: Optional[_str] = ..., copy: _bool = ...) -> DataFrame: ...
    @property
    def str(self) -> StringMethods[Series]: ...
    @property
    def dt(self) -> Series: ...
    cat = ...
    def plot(self, **kwargs) -> Union[PlotAxes, np.ndarray]: ...
    sparse = ...
    def hist(
        self,
        by: Optional[object] = ...,
        ax: Optional[PlotAxes] = ...,
        grid: _bool = ...,
        xlabelsize: Optional[int] = ...,
        xrot: Optional[float] = ...,
        ylabelsize: Optional[int] = ...,
        yrot: Optional[float] = ...,
        figsize: Optional[Tuple[float, float]] = ...,
        bins: Union[int, Sequence] = ...,
        backend: Optional[_str] = ...,
        **kwargs,
    ) -> SubplotBase: ...
    def swapaxes(
        self, axis1: SeriesAxisType, axis2: SeriesAxisType, copy: _bool = ...
    ) -> Series[S1]: ...
    def droplevel(
        self, level: Union[Level, List[Level]], axis: SeriesAxisType = ...
    ) -> DataFrame: ...
    def pop(self, item: _str) -> Series[S1]: ...
    def squeeze(self, axis: Optional[SeriesAxisType] = ...) -> Scalar: ...
    def __abs__(self) -> Series[S1]: ...
    def add_prefix(self, prefix: _str) -> Series[S1]: ...
    def add_suffix(self, suffix: _str) -> Series[S1]: ...
    def reindex(self, index: Optional[_ListLike] = ..., **kwargs) -> Series[S1]: ...
    def filter(
        self,
        items: Optional[_ListLike] = ...,
        like: Optional[_str] = ...,
        regex: Optional[_str] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def head(self, n: int = ...) -> Series[S1]: ...
    def tail(self, n: int = ...) -> Series[S1]: ...
    def sample(
        self,
        n: Optional[int] = ...,
        frac: Optional[float] = ...,
        replace: _bool = ...,
        weights: Optional[Union[_str, _ListLike, np.ndarray]] = ...,
        random_state: Optional[int] = ...,
        axis: Optional[SeriesAxisType] = ...,
        ignore_index: _bool = ...,
    ) -> Series[S1]: ...
    def astype(
        self,
        dtype: Union[S1, _str, Type[Scalar]],
        copy: _bool = ...,
        errors: Union[_str, Literal["raise", "ignore"]] = ...,
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
        axis: Optional[SeriesAxisType] = ...,
        *,
        inplace: Literal[True],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        axis: Optional[SeriesAxisType] = ...,
        *,
        inplace: Literal[False],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        axis: Optional[SeriesAxisType] = ...,
        *,
        inplace: Literal[True],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        axis: Optional[SeriesAxisType] = ...,
        *,
        inplace: Literal[False],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> Series[S1]: ...
    @overload
    def bfill(
        self,
        value: Union[S1, Dict, Series[S1], DataFrame],
        axis: SeriesAxisType = ...,
        inplace: _bool = ...,
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> Union[Series[S1], None]: ...
    def interpolate(
        self,
        method: Union[
            _str,
            Literal[
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
            ],
        ] = ...,
        axis: Optional[SeriesAxisType] = ...,
        limit: Optional[int] = ...,
        inplace: _bool = ...,
        limit_direction: Optional[
            Union[_str, Literal["forward", "backward", "both"]]
        ] = ...,
        limit_area: Optional[Union[_str, Literal["inside", "outside"]]] = ...,
        downcast: Optional[Union[_str, Literal["infer"]]] = ...,
        **kwargs,
    ) -> Series[S1]: ...
    def asof(
        self,
        where: Union[Scalar, Sequence[Scalar]],
        subset: Optional[Union[_str, Sequence[_str]]] = ...,
    ) -> Union[Scalar, Series[S1]]: ...
    def clip(
        self,
        lower: Optional[float] = ...,
        upper: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
        inplace: _bool = ...,
        *args,
        **kwargs,
    ) -> Series[S1]: ...
    def asfreq(
        self,
        freq,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill"]]
        ] = ...,
        how: Optional[Union[_str, Literal["start", "end"]]] = ...,
        normalize: _bool = ...,
        fill_value: Optional[Scalar] = ...,
    ) -> Series[S1]: ...
    def at_time(
        self,
        time: Union[_str, time],
        asof: _bool = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def between_time(
        self,
        start_time: Union[_str, time],
        end_time: Union[_str, time],
        include_start: _bool = ...,
        include_end: _bool = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def resample(
        self,
        rule,
        axis: SeriesAxisType = ...,
        closed: Optional[_str] = ...,
        label: Optional[_str] = ...,
        convention: Union[_str, Literal["start", "end", "s", "e"]] = ...,
        kind: Optional[Union[_str, Literal["timestamp", "period"]]] = ...,
        loffset=...,
        base: int = ...,
        on: Optional[_str] = ...,
        level: Optional[Level] = ...,
        origin: Union[
            Timestamp, Literal["epoch", "start", "start_day", "end", "end_day"]
        ] = ...,
        offset: Optional[Union[Timedelta, _str]] = None,
    ) -> Resampler: ...
    def first(self, offset) -> Series[S1]: ...
    def last(self, offset) -> Series[S1]: ...
    def rank(
        self,
        axis: SeriesAxisType = ...,
        method: Union[_str, Literal["average", "min", "max", "first", "dense"]] = ...,
        numeric_only: Optional[_bool] = ...,
        na_option: Union[_str, Literal["keep", "top", "bottom"]] = ...,
        ascending: _bool = ...,
        pct: _bool = ...,
    ) -> Series: ...
    def where(
        self,
        cond: Union[Series[S1], Series[_bool], np.ndarray],
        other=...,
        inplace: _bool = ...,
        axis: Optional[SeriesAxisType] = ...,
        level: Optional[Level] = ...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ) -> Series[S1]: ...
    def mask(
        self,
        cond: MaskType,
        other: Union[Scalar, Series[S1], DataFrame, Callable] = ...,
        inplace: _bool = ...,
        axis: Optional[SeriesAxisType] = ...,
        level: Optional[Level] = ...,
        errors: Union[_str, Literal["raise", "ignore"]] = ...,
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
        before: Optional[Union[date, _str, int]] = ...,
        after: Optional[Union[date, _str, int]] = ...,
        axis: Optional[SeriesAxisType] = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def tz_convert(
        self,
        tz,
        axis: SeriesAxisType = ...,
        level: Optional[Level] = ...,
        copy: _bool = ...,
    ) -> Series[S1]: ...
    def tz_localize(
        self,
        tz,
        axis: SeriesAxisType = ...,
        level: Optional[Level] = ...,
        copy: _bool = ...,
        ambiguous=...,
        nonexistent: _str = ...,
    ) -> Series[S1]: ...
    def abs(self) -> Series[S1]: ...
    def describe(
        self,
        percentiles: Optional[List[float]] = ...,
        include: Optional[Union[_str, Literal["all"], List[S1]]] = ...,
        exclude: Optional[Union[S1, List[S1]]] = ...,
        datetime_is_numeric: Optional[_bool] = ...,
    ) -> Series[S1]: ...
    def pct_change(
        self,
        periods: int = ...,
        fill_method: _str = ...,
        limit: Optional[int] = ...,
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
        bins: Optional[int] = ...,
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
        self, other: Union[num, _str, Timedelta, _ListLike, Series[S1]]
    ) -> Series: ...
    def __and__(self, other: Union[_ListLike, Series[S1]]) -> Series[_bool]: ...
    # def __array__(self, dtype: Optional[_bool] = ...) -> _np_ndarray
    def __div__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __eq__(self, other: object) -> Series[_bool]: ...  # type: ignore
    def __floordiv__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[int]: ...
    def __ge__(self, other: Union[num, _ListLike, Series[S1],Timestamp]) -> Series[_bool]: ...
    def __gt__(self, other: Union[num, _ListLike, Series[S1],Timestamp]) -> Series[_bool]: ...
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
    def __le__(self, other: Union[num, _ListLike, Series[S1],Timestamp]) -> Series[_bool]: ...
    def __lt__(self, other: Union[num, _ListLike, Series[S1],Timestamp]) -> Series[_bool]: ...
    @overload
    def __mul__(self, other: Union[Timedelta, TimedeltaSeries]) -> TimedeltaSeries: ...
    @overload
    def __mul__(self, other: Union[num, _ListLike, Series]) -> Series: ...
    def __mod__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __ne__(self, other: object) -> Series[_bool]: ...  # type: ignore
    def __pow__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __or__(self, other: Union[_ListLike, Series[S1]]) -> Series[_bool]: ...
    def __radd__(
        self, other: Union[num, _str, _ListLike, Series[S1]]
    ) -> Series[S1]: ...
    def __rand__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[_bool]: ...
    def __rdiv__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __rdivmod__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __rfloordiv__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __rmod__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __rmul__(self, other: Union[num, _ListLike, Series]) -> Series: ...
    def __rnatmul__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __rpow__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[S1]: ...
    def __ror__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[_bool]: ...
    def __rsub__(self, other: Union[num, _ListLike, Series[S1]]) -> Series: ...
    @overload
    def __rtruediv__(
        self, other: Union[Timedelta, TimedeltaSeries]
    ) -> Series[float]: ...
    @overload
    def __rtruediv__(self, other: Union[num, _ListLike, Series[S1]]) -> Series: ...
    def __rxor__(self, other: Union[num, _ListLike, Series[S1]]) -> Series[_bool]: ...
    @overload
    def __sub__(self, other: Union[Timestamp, TimestampSeries]) -> TimedeltaSeries: ...
    @overload
    def __sub__(
        self, other: Union[Timedelta, TimedeltaSeries, TimedeltaIndex]
    ) -> TimestampSeries: ...
    @overload
    def __sub__(self, other: Union[num, _ListLike, Series]) -> Series: ...
    @overload
    def __truediv__(
        self, other: Union[Timedelta, TimedeltaSeries, TimedeltaIndex]
    ) -> Series[float]: ...
    @overload
    def __truediv__(self, other: Union[num, _ListLike, Series[S1]]) -> Series: ...
    def __xor__(self, other: Union[_ListLike, Series[S1]]) -> Series: ...
    def __invert__(self) -> Series[bool]: ...
    # properties
    # @property
    # def array(self) -> _npndarray
    @property
    def at(self) -> _LocIndexerSeries[S1]: ...
    # @property
    # def cat(self) -> ?
    @property
    def iat(self) -> _iLocIndexerSeries[S1]: ...
    @property
    def iloc(self) -> _iLocIndexerSeries[S1]: ...
    @property
    def loc(self) -> _LocIndexerSeries[S1]: ...
    # Methods
    def add(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: int = ...,
    ) -> Series[S1]: ...
    def all(
        self,
        axis: SeriesAxisType = ...,
        bool_only: Optional[_bool] = ...,
        skipna: _bool = ...,
        level: Optional[Level] = ...,
        **kwargs,
    ) -> _bool: ...
    def any(
        self,
        axis: SeriesAxisType = ...,
        bool_only: Optional[_bool] = ...,
        skipna: _bool = ...,
        level: Optional[Level] = ...,
        **kwargs,
    ) -> _bool: ...
    def cummax(
        self, axis: Optional[SeriesAxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def cummin(
        self, axis: Optional[SeriesAxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def cumprod(
        self, axis: Optional[SeriesAxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def cumsum(
        self, axis: Optional[SeriesAxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> Series[S1]: ...
    def divide(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[float]: ...
    def divmod(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def eq(
        self,
        other: Union[Scalar, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def ewm(
        self,
        com: Optional[float] = ...,
        span: Optional[float] = ...,
        halflife: Optional[float] = ...,
        alpha: Optional[float] = ...,
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
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[int]: ...
    def ge(
        self,
        other: Union[Scalar, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def gt(
        self,
        other: Union[Scalar, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def item(self) -> S1: ...
    @overload
    def kurt(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def kurt(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def kurtosis(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Optional[Level],
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def kurtosis(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Scalar: ...
    def le(
        self,
        other: Union[Scalar, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def lt(
        self,
        other: Union[Scalar, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    @overload
    def mad(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def mad(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        level: None = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def max(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def max(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        *,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> S1: ...
    @overload
    def mean(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def mean(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> float: ...
    @overload
    def median(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def median(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> float: ...
    @overload
    def min(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def min(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> S1: ...
    def mod(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def mul(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def multiply(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def ne(
        self,
        other: Union[Scalar, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[_bool]: ...
    def nunique(self, dropna: _bool = ...) -> int: ...
    def pow(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    @overload
    def prod(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def prod(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def product(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def product(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Scalar: ...
    def radd(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rdivmod(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rfloordiv(
        self,
        other,
        level: Optional[Level] = ...,
        fill_value: Optional[Union[float, None]] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rmod(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rmul(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    @overload
    def rolling(
        self,
        window,
        min_periods: Optional[int] = ...,
        center: _bool = ...,
        *,
        win_type: _str,
        on: Optional[_str] = ...,
        axis: SeriesAxisType = ...,
        closed: Optional[_str] = ...,
    ) -> Window: ...
    @overload
    def rolling(
        self,
        window,
        min_periods: Optional[int] = ...,
        center: _bool = ...,
        *,
        on: Optional[_str] = ...,
        axis: SeriesAxisType = ...,
        closed: Optional[_str] = ...,
    ) -> Rolling: ...
    def rpow(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rsub(
        self,
        other: Union[Series[S1], Scalar],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    def rtruediv(
        self,
        other,
        level: Optional[Level] = ...,
        fill_value: Optional[Union[float, None]] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[S1]: ...
    @overload
    def sem(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def sem(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def skew(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def skew(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def std(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[float]: ...
    @overload
    def std(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> float: ...
    def sub(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def subtract(
        self,
        other: Union[num, _ListLike, Series[S1]],
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
        axis: Optional[SeriesAxisType] = ...,
    ) -> Series[S1]: ...
    def sum(
        self: Series[S1],
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: Optional[Level] = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        **kwargs,
    ) -> S1: ...
    def to_list(self) -> List: ...
    def to_numpy(
        self,
        dtype: Optional[Type[DtypeNp]] = ...,
        copy: _bool = ...,
        na_value=...,
        **kwargs,
    ) -> np.ndarray: ...
    def to_records(
        self,
        index: _bool = ...,
        columnDTypes: Optional[Union[_str, Dict]] = ...,
        indexDTypes: Optional[Union[_str, Dict]] = ...,
    ): ...
    def tolist(self) -> List: ...
    def truediv(
        self,
        other,
        level: Optional[Level] = ...,
        fill_value: Optional[Union[float, None]] = ...,
        axis: SeriesAxisType = ...,
    ) -> Series[float]: ...
    @overload
    def var(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> Series[S1]: ...
    @overload
    def var(
        self,
        axis: Optional[SeriesAxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Scalar: ...
    @overload
    def rename_axis(
        self,
        mapper: Union[Scalar, ListLike] = ...,
        index: Optional[Union[Scalar, ListLike, Callable, Dict]] = ...,
        columns: Optional[Union[Scalar, ListLike, Callable, Dict]] = ...,
        axis: Optional[SeriesAxisType] = ...,
        copy: _bool = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def rename_axis(
        self,
        mapper: Union[Scalar, ListLike] = ...,
        index: Optional[Union[Scalar, ListLike, Callable, Dict]] = ...,
        columns: Optional[Union[Scalar, ListLike, Callable, Dict]] = ...,
        axis: Optional[SeriesAxisType] = ...,
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
    ) -> Optional[Series[S1]]: ...

class TimestampSeries(Series): ...

class TimedeltaSeries(Series):
    # ignore needed because of mypy
    def __mul__(self, other: num) -> TimedeltaSeries: ...  # type: ignore[override]
    def __sub__(  # type: ignore[override]
        self, other: Union[Timedelta, TimedeltaSeries, TimedeltaIndex]
    ) -> TimedeltaSeries: ...
