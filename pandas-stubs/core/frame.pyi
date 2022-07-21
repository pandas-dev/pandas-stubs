from __future__ import annotations

import datetime
import datetime as _dt
from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Pattern,
    Sequence,
    Union,
    overload,
)

from matplotlib.axes import Axes as PlotAxes
import numpy as np
from pandas import (
    Timedelta,
    Timestamp,
)
from pandas.core.arraylike import OpsMixin
from pandas.core.generic import NDFrame
from pandas.core.groupby.generic import (
    _DataFrameGroupByNonScalar,
    _DataFrameGroupByScalar,
)
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index
from pandas.core.indexing import (
    _iLocIndexer,
    _LocIndexer,
)
from pandas.core.resample import Resampler
from pandas.core.series import Series
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
    FilePathOrBytesBuffer,
    GroupByObjectNonScalar,
    HashableT,
    IgnoreRaise,
    IndexingInt,
    IndexLabel,
    IndexType,
    Label,
    Level,
    MaskType,
    Renamer,
    Scalar,
    ScalarT,
    SeriesAxisType,
    StrLike,
    T as TType,
    np_ndarray_bool,
    np_ndarray_str,
    num,
)

from pandas.io.formats import format as fmt
from pandas.io.formats.style import Styler
from pandas.plotting import PlotAccessor

_str = str
_bool = bool

class _iLocIndexerFrame(_iLocIndexer):
    @overload
    def __getitem__(self, idx: tuple[int, int]) -> Scalar: ...
    @overload
    def __getitem__(self, idx: IndexingInt) -> Series: ...
    @overload
    def __getitem__(self, idx: tuple[IndexType | MaskType, int]) -> Series: ...
    @overload
    def __getitem__(self, idx: tuple[int, IndexType | MaskType]) -> Series: ...
    @overload
    def __getitem__(
        self,
        idx: IndexType | MaskType | tuple[IndexType | MaskType, IndexType | MaskType],
    ) -> DataFrame: ...
    def __setitem__(
        self,
        idx: int
        | IndexType
        | tuple[int, int]
        | tuple[IndexType, int]
        | tuple[IndexType, IndexType]
        | tuple[int, IndexType],
        value: float | Series | DataFrame | np.ndarray,
    ) -> None: ...

class _LocIndexerFrame(_LocIndexer):
    @overload
    def __getitem__(
        self,
        idx: IndexType
        | MaskType
        | list[StrLike]
        | tuple[
            IndexType
            | MaskType
            | slice
            | list[StrLike]
            | tuple[str | int | slice, ...],
            list[StrLike] | slice | Series[bool] | Callable,
        ],
    ) -> DataFrame: ...
    @overload
    def __getitem__(
        self,
        idx: tuple[int | StrLike | tuple[StrLike, ...], StrLike],
    ) -> Scalar: ...
    @overload
    def __getitem__(
        self,
        idx: ScalarT
        | tuple[IndexType | MaskType | tuple[slice, ...], ScalarT | None]
        | None,
    ) -> Series: ...
    @overload
    def __setitem__(
        self,
        idx: MaskType
        | StrLike
        | tuple[MaskType | Index | Sequence[Scalar] | Scalar | slice, ...],
        value: S1 | ArrayLike | Series | DataFrame,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: tuple[tuple[StrLike | Scalar | slice, ...], StrLike],
        value: S1 | ArrayLike | Series[S1] | list,
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: tuple[tuple[StrLike | Scalar | slice, ...], StrLike],
        value: S1 | ArrayLike | Series[S1] | list,
    ) -> None: ...

class DataFrame(NDFrame, OpsMixin):
    _ListLike = Union[
        np.ndarray,
        list[Dtype],
        dict[_str, np.ndarray],
        Sequence,
        Index,
        Series,
    ]
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __new__(
        cls,
        data: _ListLike | DataFrame | dict[Any, Any] | None = ...,
        index: Axes | None = ...,
        columns: Axes | None = ...,
        dtype=...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    @property
    def axes(self) -> list[Index]: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def style(self) -> Styler: ...
    def items(self) -> Iterable[tuple[Hashable, Series]]: ...
    def iteritems(self) -> Iterable[tuple[Label, Series]]: ...
    def iterrows(self) -> Iterable[tuple[Label, Series]]: ...
    def itertuples(self, index: _bool = ..., name: _str | None = ...): ...
    def __len__(self) -> int: ...
    @overload
    def dot(self, other: DataFrame | ArrayLike) -> DataFrame: ...
    @overload
    def dot(self, other: Series) -> Series: ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...
    @classmethod
    def from_dict(
        cls,
        data: dict[Any, Any],
        orient: Literal["columns", "index", "tight"] = ...,
        dtype: _str = ...,
        columns: list[_str] = ...,
    ) -> DataFrame: ...
    def to_numpy(
        self,
        dtype: type[DtypeNp] | Dtype | None = ...,
        copy: _bool = ...,
        na_value: Any | None = ...,
    ) -> np.ndarray: ...
    @overload
    def to_dict(
        self,
        orient: Literal["records"],
        into: Hashable = ...,
    ) -> list[dict[_str, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["dict", "list", "series", "split", "tight", "index"] = ...,
        into: Hashable = ...,
    ) -> dict[_str, Any]: ...
    def to_gbq(
        self,
        destination_table,
        project_id=...,
        chunksize=...,
        reauth=...,
        if_exists=...,
        auth_local_webserver=...,
        table_schema=...,
        location=...,
        progress_bar=...,
        credentials=...,
    ) -> None: ...
    @classmethod
    def from_records(
        cls, data, index=..., exclude=..., columns=..., coerce_float=..., nrows=...
    ) -> DataFrame: ...
    def to_records(
        self,
        index: _bool = ...,
        columnDTypes: _str | dict | None = ...,
        indexDTypes: _str | dict | None = ...,
    ) -> np.recarray: ...
    def to_stata(
        self,
        path: FilePathOrBuffer,
        convert_dates: dict | None = ...,
        write_index: _bool = ...,
        byteorder: _str | Literal["<", ">", "little", "big"] | None = ...,
        time_stamp=...,
        data_label: _str | None = ...,
        variable_labels: dict | None = ...,
        version: int = ...,
        convert_strl: list[_str] | None = ...,
    ) -> None: ...
    def to_feather(self, path: FilePathOrBuffer, **kwargs) -> None: ...
    @overload
    def to_markdown(
        self, buf: FilePathOrBuffer | None, mode: _str | None = ..., **kwargs
    ) -> None: ...
    @overload
    def to_markdown(self, mode: _str | None = ..., **kwargs) -> _str: ...
    @overload
    def to_parquet(
        self,
        path: FilePathOrBytesBuffer,
        *,
        engine: _str | Literal["auto", "pyarrow", "fastparquet"] = ...,
        compression: _str | Literal["snappy", "gzip", "brotli"] = ...,
        index: _bool | None = ...,
        partition_cols: list | None = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def to_parquet(
        self,
        *,
        path: None = ...,
        engine: _str | Literal["auto", "pyarrow", "fastparquet"] = ...,
        compression: _str | Literal["snappy", "gzip", "brotli"] = ...,
        index: _bool | None = ...,
        partition_cols: list | None = ...,
        **kwargs,
    ) -> bytes: ...
    @overload
    def to_html(
        self,
        buf: FilePathOrBuffer | None,
        columns: Sequence[_str] | None = ...,
        col_space: int | list[int] | dict[_str | int, int] | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: _str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        bold_rows: _bool = ...,
        classes: _str | list | tuple | None = ...,
        escape: _bool = ...,
        notebook: _bool = ...,
        border: int | None = ...,
        table_id: _str | None = ...,
        render_links: _bool = ...,
        encoding: _str | None = ...,
    ) -> None: ...
    @overload
    def to_html(
        self,
        columns: Sequence[_str] | None = ...,
        col_space: int | list[int] | dict[_str | int, int] | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        justify: _str | None = ...,
        max_rows: int | None = ...,
        max_cols: int | None = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        bold_rows: _bool = ...,
        classes: _str | list | tuple | None = ...,
        escape: _bool = ...,
        notebook: _bool = ...,
        border: int | None = ...,
        table_id: _str | None = ...,
        render_links: _bool = ...,
        encoding: _str | None = ...,
    ) -> _str: ...
    def info(
        self, verbose=..., buf=..., max_cols=..., memory_usage=..., null_counts=...
    ) -> None: ...
    def memory_usage(self, index: _bool = ..., deep: _bool = ...) -> Series: ...
    def transpose(self, *args, copy: _bool = ...) -> DataFrame: ...
    @property
    def T(self) -> DataFrame: ...
    @overload
    def __getitem__(self, idx: Scalar) -> Series: ...
    @overload
    def __getitem__(self, rows: slice) -> DataFrame: ...
    @overload
    def __getitem__(
        self,
        idx: tuple
        | Series[_bool]
        | DataFrame
        | list[_str]
        | list[ScalarT]
        | Index
        | np_ndarray_str
        | np_ndarray_bool
        | Sequence[tuple[Scalar, ...]],
    ) -> DataFrame: ...
    def __setitem__(self, key, value): ...
    @overload
    def query(self, expr: _str, *, inplace: Literal[True], **kwargs) -> None: ...
    @overload
    def query(
        self, expr: _str, *, inplace: Literal[False] = ..., **kwargs
    ) -> DataFrame: ...
    def eval(self, expr: _str, inplace: _bool = ..., **kwargs): ...
    def select_dtypes(
        self,
        include: _str | list[_str] | None = ...,
        exclude: _str | list[_str] | None = ...,
    ) -> DataFrame: ...
    def insert(
        self,
        loc: int,
        column,
        value: int | _ListLike,
        allow_duplicates: _bool = ...,
    ) -> None: ...
    def assign(self, **kwargs) -> DataFrame: ...
    def lookup(self, row_labels: Sequence, col_labels: Sequence) -> np.ndarray: ...
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
        fill_axis: AxisType = ...,
        broadcast_axis: AxisType | None = ...,
    ) -> DataFrame: ...
    def reindex(
        self,
        labels: Axes | None = ...,
        index: Axes | None = ...,
        columns: Axes | None = ...,
        axis: AxisType | None = ...,
        method: Literal["backfill", "bfill", "pad", "ffill", "nearest"] | None = ...,
        copy: bool = ...,
        level: int | _str = ...,
        fill_value: Scalar | None = ...,
        limit: int | None = ...,
        tolerance: float | None = ...,
    ) -> DataFrame: ...
    @overload
    def drop(
        self,
        labels: Hashable | Sequence[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] = ...,
        columns: Hashable | Sequence[Hashable] | Index = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: Hashable | Sequence[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] = ...,
        columns: Hashable | Sequence[Hashable] | Index = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame: ...
    @overload
    def drop(
        self,
        labels: Hashable | Sequence[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] = ...,
        columns: Hashable | Sequence[Hashable] | Index = ...,
        level: Level | None = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame | None: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame: ...
    @overload
    def rename(
        self,
        mapper: Renamer | None = ...,
        *,
        index: Renamer | None = ...,
        columns: Renamer | None = ...,
        axis: Axis | None = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Level | None = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame | None: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series | DataFrame | None = ...,
        method: Literal["backfill", "bfill", "ffill", "pad"] | None = ...,
        axis: AxisType | None = ...,
        limit: int = ...,
        downcast: dict | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series | DataFrame | None = ...,
        method: Literal["backfill", "bfill", "ffill", "pad"] | None = ...,
        axis: AxisType | None = ...,
        limit: int = ...,
        downcast: dict | None = ...,
        *,
        inplace: Literal[False] = ...,
    ) -> DataFrame: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series | DataFrame | None = ...,
        method: _str | Literal["backfill", "bfill", "ffill", "pad"] | None = ...,
        axis: AxisType | None = ...,
        *,
        limit: int = ...,
        downcast: dict | None = ...,
    ) -> DataFrame | None: ...
    @overload
    def fillna(
        self,
        value: Scalar | dict | Series | DataFrame | None = ...,
        method: _str | Literal["backfill", "bfill", "ffill", "pad"] | None = ...,
        axis: AxisType | None = ...,
        inplace: _bool | None = ...,
        limit: int = ...,
        downcast: dict | None = ...,
    ) -> DataFrame | None: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Scalar | Sequence | Mapping | Pattern | None = ...,
        limit: int | None = ...,
        regex=...,
        method: _str | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Scalar | Sequence | Mapping | Pattern | None = ...,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        regex=...,
        method: _str | None = ...,
    ) -> DataFrame: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Scalar | Sequence | Mapping | Pattern | None = ...,
        inplace: _bool | None = ...,
        limit: int | None = ...,
        regex=...,
        method: _str | None = ...,
    ) -> DataFrame | None: ...
    def shift(
        self,
        periods: int = ...,
        freq=...,
        axis: AxisType = ...,
        fill_value: Hashable | None = ...,
    ) -> DataFrame: ...
    @overload
    def set_index(
        self,
        keys: Label
        | Series
        | Index
        | np.ndarray
        | Iterator[HashableT]
        | list[HashableT],
        drop: _bool = ...,
        append: _bool = ...,
        verify_integrity: _bool = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def set_index(
        self,
        keys: Label
        | Series
        | Index
        | np.ndarray
        | Iterator[HashableT]
        | list[HashableT],
        drop: _bool = ...,
        append: _bool = ...,
        verify_integrity: _bool = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def set_index(
        self,
        keys: Label
        | Series
        | Index
        | np.ndarray
        | Iterator[HashableT]
        | list[HashableT],
        drop: _bool = ...,
        append: _bool = ...,
        *,
        verify_integrity: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def set_index(
        self,
        keys: Label
        | Series
        | Index
        | np.ndarray
        | Iterator[HashableT]
        | list[HashableT],
        drop: _bool = ...,
        append: _bool = ...,
        inplace: _bool | None = ...,
        verify_integrity: _bool = ...,
    ) -> DataFrame | None: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = ...,
        drop: _bool = ...,
        col_level: int | _str = ...,
        col_fill: Hashable = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = ...,
        drop: _bool = ...,
        col_level: int | _str = ...,
        col_fill: Hashable = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = ...,
        drop: _bool = ...,
        *,
        col_level: int | _str = ...,
        col_fill: Hashable = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Level | Sequence[Level] = ...,
        drop: _bool = ...,
        inplace: _bool | None = ...,
        col_level: int | _str = ...,
        col_fill: Hashable = ...,
    ) -> DataFrame | None: ...
    def isna(self) -> DataFrame: ...
    def isnull(self) -> DataFrame: ...
    def notna(self) -> DataFrame: ...
    def notnull(self) -> DataFrame: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: _str | Literal["any", "all"] = ...,
        thresh: int | None = ...,
        subset: list | None = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: _str | Literal["any", "all"] = ...,
        thresh: int | None = ...,
        subset: list | None = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: _str | Literal["any", "all"] = ...,
        thresh: int | None = ...,
        subset: list | None = ...,
    ) -> DataFrame: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: _str | Literal["any", "all"] = ...,
        thresh: int | None = ...,
        subset: list | None = ...,
        inplace: _bool | None = ...,
    ) -> DataFrame | None: ...
    def drop_duplicates(
        self,
        subset=...,
        keep: _str | Literal["first", "last"] | _bool = ...,
        inplace: _bool = ...,
        ignore_index: _bool = ...,
    ) -> DataFrame: ...
    def duplicated(
        self,
        subset: Hashable | Sequence[Hashable] | None = ...,
        keep: _str | Literal["first", "last"] | _bool = ...,
    ) -> Series: ...
    @overload
    def sort_values(
        self,
        by: _str | Sequence[_str],
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
        by: _str | Sequence[_str],
        axis: AxisType = ...,
        ascending: _bool | Sequence[_bool] = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[False],
        key: Callable | None = ...,
    ) -> DataFrame: ...
    @overload
    def sort_values(
        self,
        by: _str | Sequence[_str],
        axis: AxisType = ...,
        ascending: _bool | Sequence[_bool] = ...,
        *,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        ignore_index: _bool = ...,
        key: Callable | None = ...,
    ) -> DataFrame: ...
    @overload
    def sort_values(
        self,
        by: _str | Sequence[_str],
        axis: AxisType = ...,
        ascending: _bool | Sequence[_bool] = ...,
        inplace: _bool | None = ...,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        ignore_index: _bool = ...,
        key: Callable | None = ...,
    ) -> DataFrame | None: ...
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
        inplace: Literal[False],
        key: Callable | None = ...,
    ) -> DataFrame: ...
    @overload
    def sort_index(
        self,
        axis: AxisType = ...,
        level: Level | list[int] | list[_str] | None = ...,
        ascending: _bool | Sequence[_bool] = ...,
        *,
        kind: _str | Literal["quicksort", "mergesort", "heapsort"] = ...,
        na_position: _str | Literal["first", "last"] = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
        key: Callable | None = ...,
    ) -> DataFrame: ...
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
    ) -> DataFrame | None: ...
    def value_counts(
        self,
        subset: Sequence[Hashable] | None = ...,
        normalize: _bool = ...,
        sort: _bool = ...,
        ascending: _bool = ...,
        dropna: _bool = ...,
    ) -> Series[int]: ...
    def nlargest(
        self,
        n: int,
        columns: _str | list[_str],
        keep: _str | Literal["first", "last", "all"] = ...,
    ) -> DataFrame: ...
    def nsmallest(
        self,
        n: int,
        columns: _str | list[_str],
        keep: _str | Literal["first", "last", "all"] = ...,
    ) -> DataFrame: ...
    def swaplevel(
        self, i: Level = ..., j: Level = ..., axis: AxisType = ...
    ) -> DataFrame: ...
    def reorder_levels(self, order: list, axis: AxisType = ...) -> DataFrame: ...
    def compare(
        self,
        other: DataFrame,
        align_axis: Axis = ...,
        keep_shape: bool = ...,
        keep_equal: bool = ...,
    ) -> DataFrame: ...
    def combine(
        self,
        other: DataFrame,
        func: Callable,
        fill_value=...,
        overwrite: _bool = ...,
    ) -> DataFrame: ...
    def combine_first(self, other: DataFrame) -> DataFrame: ...
    def update(
        self,
        other: DataFrame | Series,
        join: _str = ...,
        overwrite: _bool = ...,
        filter_func: Callable | None = ...,
        errors: _str | Literal["raise", "ignore"] = ...,
    ) -> None: ...
    @overload
    def groupby(
        self,
        by: Scalar,
        axis: AxisType = ...,
        level: Level | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        squeeze: _bool = ...,
        observed: _bool = ...,
        dropna: _bool = ...,
    ) -> _DataFrameGroupByScalar: ...
    @overload
    def groupby(
        self,
        by: GroupByObjectNonScalar | None = ...,
        axis: AxisType = ...,
        level: Level | None = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        squeeze: _bool = ...,
        observed: _bool = ...,
        dropna: _bool = ...,
    ) -> _DataFrameGroupByNonScalar: ...
    def pivot(
        self,
        index=...,
        columns=...,
        values=...,
    ) -> DataFrame: ...
    def pivot_table(
        self,
        values: _str | None = ...,
        index: _str | Grouper | Sequence | None = ...,
        columns: _str | Grouper | Sequence | None = ...,
        aggfunc=...,
        fill_value: Scalar | None = ...,
        margins: _bool = ...,
        dropna: _bool = ...,
        margins_name: _str = ...,
        observed: _bool = ...,
    ) -> DataFrame: ...
    def stack(self, level: Level = ..., dropna: _bool = ...) -> DataFrame | Series: ...
    def explode(self, column: _str | tuple, ignore_index: _bool = ...) -> DataFrame: ...
    def unstack(
        self,
        level: Level = ...,
        fill_value: int | _str | dict | None = ...,
    ) -> DataFrame | Series: ...
    def melt(
        self,
        id_vars: tuple | Sequence | np.ndarray | None = ...,
        value_vars: tuple | Sequence | np.ndarray | None = ...,
        var_name: Scalar | None = ...,
        value_name: Scalar = ...,
        col_level: int | _str | None = ...,
        ignore_index: _bool = ...,
    ) -> DataFrame: ...
    def diff(self, periods: int = ..., axis: AxisType = ...) -> DataFrame: ...
    @overload
    def agg(self, func: Callable | _str, axis: AxisType = ..., **kwargs) -> Series: ...
    @overload
    def agg(
        self,
        func: list[Callable] | dict[_str, Callable] = ...,
        axis: AxisType = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self, func: Callable | _str, axis: AxisType = ..., **kwargs
    ) -> Series: ...
    @overload
    def aggregate(
        self,
        func: list[Callable] | dict[_str, Callable],
        axis: AxisType = ...,
        **kwargs,
    ) -> DataFrame: ...
    def transform(
        self,
        func: list[Callable] | dict[_str, Callable],
        axis: AxisType = ...,
        *args,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def apply(self, f: Callable) -> Series: ...
    @overload
    def apply(
        self,
        f: Callable,
        axis: AxisType,
        raw: _bool = ...,
        result_type: _str | None = ...,
        args=...,
        **kwargs,
    ) -> DataFrame: ...
    def applymap(
        self, func: Callable, na_action: Literal["ignore"] | None = ..., **kwargs
    ) -> DataFrame: ...
    def append(
        self,
        other: DataFrame
        | Series
        | dict[Any, Any]
        | Sequence[Scalar]
        | Sequence[_ListLike],
        ignore_index: _bool = ...,
        verify_integrity: _bool = ...,
        sort: _bool = ...,
    ) -> DataFrame: ...
    def join(
        self,
        other: DataFrame | Series | list[DataFrame | Series],
        on: _str | list[_str] | None = ...,
        how: _str | Literal["left", "right", "outer", "inner"] = ...,
        lsuffix: _str = ...,
        rsuffix: _str = ...,
        sort: _bool = ...,
    ) -> DataFrame: ...
    def merge(
        self,
        right: DataFrame | Series,
        how: _str | Literal["left", "right", "inner", "outer"] = ...,
        on: IndexLabel | None = ...,
        left_on: IndexLabel | None = ...,
        right_on: IndexLabel | None = ...,
        left_index: _bool = ...,
        right_index: _bool = ...,
        sort: _bool = ...,
        suffixes: tuple[_str | None, _str | None] = ...,
        copy: _bool = ...,
        indicator: _bool | _str = ...,
        validate: _str | None = ...,
    ) -> DataFrame: ...
    def round(
        self, decimals: int | dict | Series = ..., *args, **kwargs
    ) -> DataFrame: ...
    def corr(
        self,
        method: _str | Literal["pearson", "kendall", "spearman"] = ...,
        min_periods: int = ...,
    ) -> DataFrame: ...
    def cov(self, min_periods: int | None = ..., ddof: int = ...) -> DataFrame: ...
    def corrwith(
        self,
        other: DataFrame | Series,
        axis: AxisType | None = ...,
        drop: _bool = ...,
        method: _str | Literal["pearson", "kendall", "spearman"] = ...,
    ) -> Series: ...
    @overload
    def count(
        self, axis: AxisType = ..., numeric_only: _bool = ..., *, level: Level
    ) -> DataFrame: ...
    @overload
    def count(
        self, axis: AxisType = ..., level: None = ..., numeric_only: _bool = ...
    ) -> Series: ...
    def nunique(self, axis: AxisType = ..., dropna: bool = ...) -> Series: ...
    def idxmax(self, axis: AxisType = ..., skipna: _bool = ...) -> Series: ...
    def idxmin(self, axis: AxisType = ..., skipna: _bool = ...) -> Series: ...
    @overload
    def mode(
        self,
        axis: AxisType = ...,
        skipna: _bool = ...,
        numeric_only: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def mode(
        self,
        axis: AxisType = ...,
        skipna: _bool = ...,
        level: None = ...,
        numeric_only: _bool = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: float = ...,
        axis: AxisType = ...,
        numeric_only: _bool = ...,
        interpolation: _str
        | Literal["linear", "lower", "higher", "midpoint", "nearest"] = ...,
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: list[float] | np.ndarray,
        axis: AxisType = ...,
        numeric_only: _bool = ...,
        interpolation: _str
        | Literal["linear", "lower", "higher", "midpoint", "nearest"] = ...,
    ) -> DataFrame: ...
    def to_timestamp(
        self,
        freq=...,
        how: _str | Literal["start", "end", "s", "e"] = ...,
        axis: AxisType = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    def to_period(
        self, freq: _str | None = ..., axis: AxisType = ..., copy: _bool = ...
    ) -> DataFrame: ...
    def isin(self, values: Iterable | Series | DataFrame | dict) -> DataFrame: ...
    @property
    def plot(self) -> PlotAccessor: ...
    def hist(
        self,
        column: _str | list[_str] | None = ...,
        by: _str | _ListLike | None = ...,
        grid: _bool = ...,
        xlabelsize: int | None = ...,
        xrot: float | None = ...,
        ylabelsize: int | None = ...,
        yrot: float | None = ...,
        ax: PlotAxes | None = ...,
        sharex: _bool = ...,
        sharey: _bool = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        bins: int | list = ...,
        backend: _str | None = ...,
        **kwargs,
    ): ...
    def boxplot(
        self,
        column: _str | list[_str] | None = ...,
        by: _str | _ListLike | None = ...,
        ax: PlotAxes | None = ...,
        fontsize: float | _str | None = ...,
        rot: int = ...,
        grid: _bool = ...,
        figsize: tuple[float, float] | None = ...,
        layout: tuple[int, int] | None = ...,
        return_type: _str | Literal["axes", "dict", "both"] | None = ...,
        backend: _str | None = ...,
        **kwargs,
    ): ...
    sparse = ...

    # The rest of these are remnants from the
    # stubs shipped at preview. They may belong in
    # base classes, or stubgen just failed to generate
    # these.

    Name: _str
    #
    # dunder methods
    def __exp__(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType = ...,
        level: Level = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def __iter__(self) -> Iterator[int | float | _str]: ...
    # properties
    @property
    def at(self): ...  # Not sure what to do with this yet; look at source
    @property
    def bool(self) -> _bool: ...
    @property
    def columns(self) -> Index: ...
    @columns.setter  # setter needs to be right next to getter; otherwise mypy complains
    def columns(self, cols: list[_str] | Index[_str]) -> None: ...  # type: ignore[type-arg]
    @property
    def dtypes(self) -> Series: ...
    @property
    def empty(self) -> _bool: ...
    @property
    def iat(self): ...  # Not sure what to do with this yet; look at source
    @property
    def iloc(self) -> _iLocIndexerFrame: ...
    @property
    def index(self) -> Index: ...
    @index.setter
    def index(self, idx: Index) -> None: ...
    @property
    def loc(self) -> _LocIndexerFrame: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def values(self) -> np.ndarray: ...
    # methods
    def abs(self) -> DataFrame: ...
    def add(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def add_prefix(self, prefix: _str) -> DataFrame: ...
    def add_suffix(self, suffix: _str) -> DataFrame: ...
    @overload
    def all(
        self,
        axis: AxisType = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def all(
        self,
        axis: AxisType = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def any(
        self,
        axis: AxisType = ...,
        bool_only: _bool | None = ...,
        skipna: _bool = ...,
        level: None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def any(
        self,
        axis: AxisType = ...,
        bool_only: _bool = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    def asof(self, where, subset: _str | list[_str] | None = ...) -> DataFrame: ...
    def asfreq(
        self,
        freq,
        method: _str | Literal["backfill", "bfill", "pad", "ffill"] | None = ...,
        how: _str | Literal["start", "end"] | None = ...,
        normalize: _bool = ...,
        fill_value: Scalar | None = ...,
    ) -> DataFrame: ...
    def astype(
        self,
        dtype: _str | Dtype | dict[_str, _str | Dtype],
        copy: _bool = ...,
        errors: _str = ...,
    ) -> DataFrame: ...
    def at_time(
        self,
        time: _str | datetime.time,
        asof: _bool = ...,
        axis: AxisType | None = ...,
    ) -> DataFrame: ...
    def between_time(
        self,
        start_time: _str | datetime.time,
        end_time: _str | datetime.time,
        include_start: _bool = ...,
        include_end: _bool = ...,
        axis: AxisType | None = ...,
    ) -> DataFrame: ...
    @overload
    def bfill(
        self,
        axis: AxisType | None = ...,
        *,
        inplace: Literal[True],
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        axis: AxisType | None = ...,
        *,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> DataFrame: ...
    def clip(
        self,
        lower: float | None = ...,
        upper: float | None = ...,
        axis: AxisType | None = ...,
        inplace: _bool = ...,
        *args,
        **kwargs,
    ) -> DataFrame: ...
    def copy(self, deep: _bool = ...) -> DataFrame: ...
    def cummax(
        self, axis: AxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def cummin(
        self, axis: AxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def cumprod(
        self, axis: AxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def cumsum(
        self, axis: AxisType | None = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def describe(
        self,
        percentiles: list[float] | None = ...,
        include: _str | Literal["all"] | list[Dtype] | None = ...,
        exclude: list[Dtype] | None = ...,
        datetime_is_numeric: _bool | None = ...,
    ) -> DataFrame: ...
    def div(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def divide(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def droplevel(
        self, level: Level | list[Level] = ..., axis: AxisType = ...
    ) -> DataFrame: ...
    def eq(
        self, other, axis: AxisType = ..., level: Level | None = ...
    ) -> DataFrame: ...
    def equals(self, other: Series | DataFrame) -> _bool: ...
    def ewm(
        self,
        com: float | None = ...,
        span: float | None = ...,
        halflife: float | None = ...,
        alpha: float | None = ...,
        min_periods: int = ...,
        adjust: _bool = ...,
        ignore_na: _bool = ...,
        axis: AxisType = ...,
    ) -> DataFrame: ...
    def exp(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def expanding(
        self, min_periods: int = ..., center: _bool = ..., axis: AxisType = ...
    ): ...  # for now
    @overload
    def ffill(
        self,
        axis: AxisType | None = ...,
        *,
        inplace: Literal[True],
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        axis: AxisType | None = ...,
        *,
        inplace: Literal[False] = ...,
        limit: int | None = ...,
        downcast: dict | None = ...,
    ) -> DataFrame: ...
    def filter(
        self,
        items: list | None = ...,
        like: _str | None = ...,
        regex: _str | None = ...,
        axis: AxisType | None = ...,
    ) -> DataFrame: ...
    def first(self, offset) -> DataFrame: ...
    def first_valid_index(self) -> Scalar: ...
    def floordiv(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    # def from_dict
    # def from_records
    def fulldiv(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def ge(
        self, other, axis: AxisType = ..., level: Level | None = ...
    ) -> DataFrame: ...
    # def get
    def gt(
        self, other, axis: AxisType = ..., level: Level | None = ...
    ) -> DataFrame: ...
    def head(self, n: int = ...) -> DataFrame: ...
    def infer_objects(self) -> DataFrame: ...
    # def info
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        limit_direction: _str | Literal["forward", "backward", "both"] = ...,
        limit_area: _str | Literal["inside", "outside"] | None = ...,
        downcast: _str | Literal["infer"] | None = ...,
        *,
        inplace: Literal[True],
        **kwargs,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        limit_direction: _str | Literal["forward", "backward", "both"] = ...,
        limit_area: _str | Literal["inside", "outside"] | None = ...,
        downcast: _str | Literal["infer"] | None = ...,
        *,
        inplace: Literal[False],
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        limit_direction: _str | Literal["forward", "backward", "both"] = ...,
        limit_area: _str | Literal["inside", "outside"] | None = ...,
        downcast: _str | Literal["infer"] | None = ...,
    ) -> DataFrame: ...
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: int | None = ...,
        inplace: _bool | None = ...,
        limit_direction: _str | Literal["forward", "backward", "both"] = ...,
        limit_area: _str | Literal["inside", "outside"] | None = ...,
        downcast: _str | Literal["infer"] | None = ...,
        **kwargs,
    ) -> DataFrame: ...
    def keys(self) -> Index: ...
    @overload
    def kurt(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def kurt(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def kurtosis(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def kurtosis(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    def last(self, offset) -> DataFrame: ...
    def last_valid_index(self) -> Scalar: ...
    def le(
        self, other, axis: AxisType = ..., level: Level | None = ...
    ) -> DataFrame: ...
    def lt(
        self, other, axis: AxisType = ..., level: Level | None = ...
    ) -> DataFrame: ...
    @overload
    def mad(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
    ) -> Series: ...
    @overload
    def mad(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    def mask(
        self,
        cond: Series | DataFrame | np.ndarray,
        other=...,
        inplace: _bool = ...,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def max(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def max(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def mean(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def mean(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def median(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def median(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def min(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def min(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    def mod(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def mul(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def multiply(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def ne(
        self, other, axis: AxisType = ..., level: Level | None = ...
    ) -> DataFrame: ...
    def pct_change(
        self,
        periods: int = ...,
        fill_method: _str = ...,
        limit: int | None = ...,
        freq=...,
        **kwargs,
    ) -> DataFrame: ...
    def pipe(
        self,
        func: Callable[..., TType] | tuple[Callable[..., TType], _str],
        *args,
        **kwargs,
    ) -> TType: ...
    def pop(self, item: _str) -> Series: ...
    def pow(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    @overload
    def prod(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def prod(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Series: ...
    def product(
        self,
        axis: AxisType | None = ...,
        skipna: _bool = ...,
        level: Level | None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        **kwargs,
    ) -> DataFrame: ...
    def radd(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def rank(
        self,
        axis: AxisType = ...,
        method: _str | Literal["average", "min", "max", "first", "dense"] = ...,
        numeric_only: _bool | None = ...,
        na_option: _str | Literal["keep", "top", "bottom"] = ...,
        ascending: _bool = ...,
        pct: _bool = ...,
    ) -> DataFrame: ...
    def rdiv(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def reindex_like(
        self,
        other: DataFrame,
        method: _str
        | Literal["backfill", "bfill", "pad", "ffill", "nearest"]
        | None = ...,
        copy: _bool = ...,
        limit: int | None = ...,
        tolerance=...,
    ) -> DataFrame: ...
    @overload
    def rename_axis(
        self,
        mapper=...,
        *,
        inplace: Literal[True],
        axis: AxisType | None = ...,
        copy: _bool = ...,
    ) -> None: ...
    @overload
    def rename_axis(
        self,
        mapper=...,
        *,
        inplace: Literal[False] = ...,
        axis: AxisType | None = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def rename_axis(
        self,
        *,
        inplace: Literal[True],
        index: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        columns: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        copy: _bool = ...,
    ) -> None: ...
    @overload
    def rename_axis(
        self,
        *,
        inplace: Literal[False] = ...,
        index: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        columns: _str | Sequence[_str] | dict[_str | int, _str] | Callable | None = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    def resample(
        self,
        rule,
        axis: AxisType = ...,
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
    def rfloordiv(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def rmod(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def rmul(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    @overload
    def rolling(
        self,
        window,
        min_periods: int | None = ...,
        center: _bool = ...,
        *,
        win_type: _str,
        on: _str | None = ...,
        axis: AxisType = ...,
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
        axis: AxisType = ...,
        closed: _str | None = ...,
    ) -> Rolling: ...
    def rpow(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def rsub(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def rtruediv(
        self,
        other,
        axis: AxisType = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    # sample is missing a weights arg
    def sample(
        self,
        n: int | None = ...,
        frac: float | None = ...,
        replace: _bool = ...,
        weights: _str | _ListLike | np.ndarray | None = ...,
        random_state: int | None = ...,
        axis: SeriesAxisType | None = ...,
        ignore_index: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def sem(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def sem(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def set_axis(
        self, labels, inplace: Literal[True], axis: AxisType = ...
    ) -> None: ...
    @overload
    def set_axis(
        self, labels, inplace: Literal[False], axis: AxisType = ...
    ) -> DataFrame: ...
    @overload
    def set_axis(self, labels, *, axis: AxisType = ...) -> DataFrame: ...
    @overload
    def set_axis(
        self,
        labels,
        axis: AxisType = ...,
        inplace: _bool | None = ...,
    ) -> DataFrame | None: ...
    @overload
    def skew(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    def slice_shift(self, periods: int = ..., axis: AxisType = ...) -> DataFrame: ...
    def squeeze(self, axis: AxisType | None = ...): ...
    @overload
    def std(
        self,
        axis: AxisType = ...,
        skipna: _bool = ...,
        ddof: int = ...,
        numeric_only: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def std(
        self,
        axis: AxisType = ...,
        skipna: _bool = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool = ...,
        **kwargs,
    ) -> Series: ...
    def sub(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def subtract(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    @overload
    def sum(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def sum(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        numeric_only: _bool | None = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Series: ...
    def swapaxes(
        self, axis1: AxisType, axis2: AxisType, copy: _bool = ...
    ) -> DataFrame: ...
    def tail(self, n: int = ...) -> DataFrame: ...
    def take(
        self,
        indices: list,
        axis: AxisType = ...,
        is_copy: _bool | None = ...,
        **kwargs,
    ) -> DataFrame: ...
    def tshift(
        self, periods: int = ..., freq=..., axis: AxisType = ...
    ) -> DataFrame: ...
    def to_clipboard(
        self, excel: _bool = ..., sep: _str | None = ..., **kwargs
    ) -> None: ...
    @overload
    def to_csv(
        self,
        path_or_buf: FilePathOrBuffer | None,
        sep: _str = ...,
        na_rep: _str = ...,
        float_format: _str | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        index_label: _bool | _str | Sequence[Hashable] | None = ...,
        mode: _str = ...,
        encoding: _str | None = ...,
        compression: _str | Mapping[_str, _str] = ...,
        quoting: int | None = ...,
        quotechar: _str = ...,
        line_terminator: _str | None = ...,
        chunksize: int | None = ...,
        date_format: _str | None = ...,
        doublequote: _bool = ...,
        escapechar: _str | None = ...,
        decimal: _str = ...,
        errors: _str = ...,
        storage_options: dict[_str, Any] | None = ...,
    ) -> None: ...
    @overload
    def to_csv(
        self,
        sep: _str = ...,
        na_rep: _str = ...,
        float_format: _str | None = ...,
        columns: Sequence[Hashable] | None = ...,
        header: _bool | list[_str] = ...,
        index: _bool = ...,
        index_label: _bool | _str | Sequence[Hashable] | None = ...,
        mode: _str = ...,
        encoding: _str | None = ...,
        compression: _str | Mapping[_str, _str] = ...,
        quoting: int | None = ...,
        quotechar: _str = ...,
        line_terminator: _str | None = ...,
        chunksize: int | None = ...,
        date_format: _str | None = ...,
        doublequote: _bool = ...,
        escapechar: _str | None = ...,
        decimal: _str = ...,
        errors: _str = ...,
        storage_options: dict[_str, Any] | None = ...,
    ) -> _str: ...
    def to_excel(
        self,
        excel_writer,
        sheet_name: _str = ...,
        na_rep: _str = ...,
        float_format: _str | None = ...,
        columns: _str | Sequence[_str] | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        index_label: _str | Sequence[_str] | None = ...,
        startrow: int = ...,
        startcol: int = ...,
        engine: _str | None = ...,
        merge_cells: _bool = ...,
        encoding: _str | None = ...,
        inf_rep: _str = ...,
        verbose: _bool = ...,
        freeze_panes: tuple[int, int] | None = ...,
    ) -> None: ...
    def to_hdf(
        self,
        path_or_buf: FilePathOrBuffer,
        key: _str,
        mode: _str = ...,
        complevel: int | None = ...,
        complib: _str | None = ...,
        append: _bool = ...,
        format: _str | None = ...,
        index: _bool = ...,
        min_itemsize: int | dict[_str, int] | None = ...,
        nan_rep=...,
        dropna: _bool | None = ...,
        data_columns: list[_str] | None = ...,
        errors: _str = ...,
        encoding: _str = ...,
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: FilePathOrBuffer | None,
        orient: _str
        | Literal["split", "records", "index", "columns", "values", "table"]
        | None = ...,
        date_format: _str | Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: _str | Literal["s", "ms", "us", "ns"] = ...,
        default_handler: Callable[[Any], _str | int | float | _bool | list | dict]
        | None = ...,
        lines: _bool = ...,
        compression: _str | Literal["infer", "gzip", "bz2", "zip", "xz"] | None = ...,
        index: _bool = ...,
        indent: int | None = ...,
    ) -> None: ...
    @overload
    def to_json(
        self,
        orient: _str
        | Literal["split", "records", "index", "columns", "values", "table"]
        | None = ...,
        date_format: _str | Literal["epoch", "iso"] | None = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: _str | Literal["s", "ms", "us", "ns"] = ...,
        default_handler: Callable[[Any], _str | int | float | _bool | list | dict]
        | None = ...,
        lines: _bool = ...,
        compression: _str | Literal["infer", "gzip", "bz2", "zip", "xz"] | None = ...,
        index: _bool = ...,
        indent: int | None = ...,
    ) -> _str: ...
    @overload
    def to_latex(
        self,
        buf: FilePathOrBuffer | None,
        columns: list[_str] | None = ...,
        col_space: int | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        bold_rows: _bool = ...,
        column_format: _str | None = ...,
        longtable: _bool | None = ...,
        escape: _bool | None = ...,
        encoding: _str | None = ...,
        decimal: _str = ...,
        multicolumn: _bool | None = ...,
        multicolumn_format: _str | None = ...,
        multirow: _bool | None = ...,
        caption: _str | tuple[_str, _str] | None = ...,
        label: _str | None = ...,
        position: _str | None = ...,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        columns: list[_str] | None = ...,
        col_space: int | None = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: _bool | None = ...,
        index_names: _bool = ...,
        bold_rows: _bool = ...,
        column_format: _str | None = ...,
        longtable: _bool | None = ...,
        escape: _bool | None = ...,
        encoding: _str | None = ...,
        decimal: _str = ...,
        multicolumn: _bool | None = ...,
        multicolumn_format: _str | None = ...,
        multirow: _bool | None = ...,
        caption: _str | tuple[_str, _str] | None = ...,
        label: _str | None = ...,
        position: _str | None = ...,
    ) -> _str: ...
    def to_pickle(
        self,
        path: _str,
        compression: _str | Literal["infer", "gzip", "bz2", "zip", "xz"] = ...,
        protocol: int = ...,
    ) -> None: ...
    def to_sql(
        self,
        name: _str,
        con,
        schema: _str | None = ...,
        if_exists: _str = ...,
        index: _bool = ...,
        index_label: _str | Sequence[_str] | None = ...,
        chunksize: int | None = ...,
        dtype: dict | Scalar | None = ...,
        method: _str | Callable | None = ...,
    ) -> None: ...
    @overload
    def to_string(
        self,
        buf: FilePathOrBuffer | None,
        columns: Sequence[_str] | None = ...,
        col_space: int | list[int] | dict[_str | int, int] | None = ...,
        header: _bool | Sequence[_str] = ...,
        index: _bool = ...,
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
        columns: Sequence[_str] | None = ...,
        col_space: int | list[int] | dict[_str | int, int] | None = ...,
        header: _bool | Sequence[_str] = ...,
        index: _bool = ...,
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
    def to_xarray(self): ...
    def truediv(
        self,
        other: num | _ListLike | DataFrame,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        fill_value: float | None = ...,
    ) -> DataFrame: ...
    def truncate(
        self,
        before: datetime.date | _str | int | None = ...,
        after: datetime.date | _str | int | None = ...,
        axis: AxisType | None = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    # def tshift
    def tz_convert(
        self,
        tz,
        axis: AxisType = ...,
        level: Level | None = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    def tz_localize(
        self,
        tz,
        axis: AxisType = ...,
        level: Level | None = ...,
        copy: _bool = ...,
        ambiguous=...,
        nonexistent: _str = ...,
    ) -> DataFrame: ...
    def unique(self) -> DataFrame: ...
    @overload
    def var(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def var(
        self,
        axis: AxisType | None = ...,
        skipna: _bool | None = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: _bool | None = ...,
        **kwargs,
    ) -> Series: ...
    def where(
        self,
        cond: Series | DataFrame | np.ndarray,
        other=...,
        inplace: _bool = ...,
        axis: AxisType | None = ...,
        level: Level | None = ...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ) -> DataFrame: ...
    # Move from generic because Series is Generic and it returns Series[bool] there
    def __invert__(self) -> DataFrame: ...
