from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Literal,
    Mapping,
    Sequence,
    overload,
)

import numpy as np
from pandas.core.base import PandasObject
from pandas.core.indexes.base import Index
import pandas.core.indexing as indexing

from pandas._typing import (
    S1,
    ArrayLike,
    Axis,
    Dtype,
    FilePathOrBuffer,
    FrameOrSeries,
    FrameOrSeriesUnion,
    IgnoreRaise,
    Level,
    Scalar,
    SeriesAxisType,
    T,
)

_bool = bool
_str = str

class NDFrame(PandasObject, indexing.IndexingMixin):
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def set_flags(
        self: FrameOrSeries,
        *,
        copy: bool = ...,
        allows_duplicate_labels: bool | None = ...,
    ) -> FrameOrSeries: ...
    @property
    def attrs(self) -> dict[Hashable | None, Any]: ...
    @attrs.setter
    def attrs(self, value: Mapping[Hashable | None, Any]) -> None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def axes(self) -> list[Index]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    def swapaxes(
        self, axis1: SeriesAxisType, axis2: SeriesAxisType, copy: _bool = ...
    ) -> NDFrame: ...
    def droplevel(self, level: Level, axis: SeriesAxisType = ...) -> NDFrame: ...
    def pop(self, item: _str) -> NDFrame: ...
    def squeeze(self, axis=...): ...
    def swaplevel(self, i=..., j=..., axis=...) -> NDFrame: ...
    def equals(self, other: Series[S1]) -> _bool: ...
    def __neg__(self) -> None: ...
    def __pos__(self) -> None: ...
    def __nonzero__(self) -> None: ...
    def bool(self) -> _bool: ...
    def __abs__(self) -> NDFrame: ...
    def __round__(self, decimals: int = ...) -> NDFrame: ...
    def keys(self): ...
    def iteritems(self): ...
    def __len__(self) -> int: ...
    def __contains__(self, key) -> _bool: ...
    @property
    def empty(self) -> _bool: ...
    __array_priority__: int = ...
    def __array__(self, dtype=...) -> np.ndarray: ...
    def __array_wrap__(self, result, context=...): ...
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
        default_handler: Callable[[Any], _str | float | _bool | list | dict]
        | None = ...,
        lines: _bool = ...,
        compression: _str | Literal["infer", "gzip", "bz2", "zip", "xz"] = ...,
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
        default_handler: Callable[[Any], _str | float | _bool | list | dict]
        | None = ...,
        lines: _bool = ...,
        compression: _str | Literal["infer", "gzip", "bz2", "zip", "xz"] | None = ...,
        index: _bool = ...,
        indent: int | None = ...,
    ) -> _str: ...
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
    def to_pickle(
        self,
        path: _str,
        compression: _str | Literal["infer", "gzip", "bz2", "zip", "xz"] = ...,
        protocol: int = ...,
    ) -> None: ...
    def to_clipboard(
        self, excel: _bool = ..., sep: _str | None = ..., **kwargs
    ) -> None: ...
    def to_xarray(self): ...
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
    def take(
        self, indices, axis=..., is_copy: _bool | None = ..., **kwargs
    ) -> NDFrame: ...
    def xs(
        self,
        key: _str | tuple[_str],
        axis: SeriesAxisType = ...,
        level: Level | None = ...,
        drop_level: _bool = ...,
    ) -> FrameOrSeriesUnion: ...
    def __delitem__(self, idx: Hashable): ...
    def get(self, key: object, default: Dtype | None = ...) -> Dtype: ...
    def reindex_like(
        self,
        other,
        method: _str | None = ...,
        copy: _bool = ...,
        limit=...,
        tolerance=...,
    ) -> NDFrame: ...
    @overload
    def drop(
        self,
        labels: Hashable | Sequence[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] = ...,
        columns: Hashable | Sequence[Hashable] = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self: NDFrame,
        labels: Hashable | Sequence[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] = ...,
        columns: Hashable | Sequence[Hashable] = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> NDFrame: ...
    @overload
    def drop(
        self: NDFrame,
        labels: Hashable | Sequence[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | Sequence[Hashable] = ...,
        columns: Hashable | Sequence[Hashable] = ...,
        level: Level | None = ...,
        inplace: _bool = ...,
        errors: IgnoreRaise = ...,
    ) -> NDFrame | None: ...
    def add_prefix(self, prefix: _str) -> NDFrame: ...
    def add_suffix(self, suffix: _str) -> NDFrame: ...
    def sort_index(
        self,
        axis=...,
        level=...,
        ascending: _bool = ...,
        inplace: _bool = ...,
        kind: _str = ...,
        na_position: _str = ...,
        sort_remaining: _bool = ...,
        ignore_index: _bool = ...,
    ): ...
    def filter(
        self,
        items=...,
        like: _str | None = ...,
        regex: _str | None = ...,
        axis=...,
    ) -> NDFrame: ...
    def head(self: FrameOrSeries, n: int = ...) -> FrameOrSeries: ...
    def tail(self: FrameOrSeries, n: int = ...) -> FrameOrSeries: ...
    def pipe(
        self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
    ) -> T: ...
    def __finalize__(self, other, method=..., **kwargs) -> NDFrame: ...
    def __getattr__(self, name: _str): ...
    def __setattr__(self, name: _str, value) -> None: ...
    @property
    def values(self) -> ArrayLike: ...
    @property
    def dtypes(self): ...
    def astype(
        self: FrameOrSeries, dtype, copy: _bool = ..., errors: str = ...
    ) -> FrameOrSeries: ...
    def copy(self: FrameOrSeries, deep: _bool = ...) -> FrameOrSeries: ...
    def __copy__(self, deep: _bool = ...) -> NDFrame: ...
    def __deepcopy__(self, memo=...) -> NDFrame: ...
    def infer_objects(self) -> NDFrame: ...
    def convert_dtypes(
        self: FrameOrSeries,
        infer_objects: _bool = ...,
        convert_string: _bool = ...,
        convert_integer: _bool = ...,
        convert_boolean: _bool = ...,
    ) -> FrameOrSeries: ...
    def fillna(
        self,
        value=...,
        method=...,
        axis=...,
        inplace: _bool = ...,
        limit=...,
        downcast=...,
    ) -> NDFrame | None: ...
    def replace(
        self,
        to_replace=...,
        value=...,
        inplace: _bool = ...,
        limit=...,
        regex: _bool = ...,
        method: _str = ...,
    ): ...
    def asof(self, where, subset=...): ...
    def isna(self) -> NDFrame: ...
    def isnull(self) -> NDFrame: ...
    def notna(self) -> NDFrame: ...
    def notnull(self) -> NDFrame: ...
    def clip(
        self, lower=..., upper=..., axis=..., inplace: _bool = ..., *args, **kwargs
    ) -> NDFrame: ...
    def asfreq(
        self,
        freq,
        method=...,
        how: _str | None = ...,
        normalize: _bool = ...,
        fill_value=...,
    ) -> NDFrame: ...
    def at_time(self, time, asof: _bool = ..., axis=...) -> NDFrame: ...
    def between_time(
        self,
        start_time,
        end_time,
        include_start: _bool = ...,
        include_end: _bool = ...,
        axis=...,
    ) -> NDFrame: ...
    def first(self, offset) -> NDFrame: ...
    def last(self, offset) -> NDFrame: ...
    def rank(
        self,
        axis=...,
        method: _str = ...,
        numeric_only: _bool | None = ...,
        na_option: _str = ...,
        ascending: _bool = ...,
        pct: _bool = ...,
    ) -> NDFrame: ...
    def where(
        self,
        cond,
        other=...,
        inplace: _bool = ...,
        axis=...,
        level=...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ): ...
    def mask(
        self,
        cond,
        other=...,
        inplace: _bool = ...,
        axis=...,
        level=...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ): ...
    def shift(self, periods=..., freq=..., axis=..., fill_value=...) -> NDFrame: ...
    def slice_shift(self, periods: int = ..., axis=...) -> NDFrame: ...
    def tshift(self, periods: int = ..., freq=..., axis=...) -> NDFrame: ...
    def truncate(
        self, before=..., after=..., axis=..., copy: _bool = ...
    ) -> NDFrame: ...
    def tz_convert(self, tz, axis=..., level=..., copy: _bool = ...) -> NDFrame: ...
    def tz_localize(
        self,
        tz,
        axis=...,
        level=...,
        copy: _bool = ...,
        ambiguous=...,
        nonexistent: str = ...,
    ) -> NDFrame: ...
    def abs(self) -> NDFrame: ...
    def describe(
        self,
        percentiles=...,
        include=...,
        exclude=...,
        datetime_is_numeric: _bool | None = ...,
    ) -> NDFrame: ...
    def pct_change(
        self, periods=..., fill_method=..., limit=..., freq=..., **kwargs
    ) -> NDFrame: ...
    def transform(self, func, *args, **kwargs): ...
    def first_valid_index(self): ...
    def last_valid_index(self): ...

from pandas.core.series import Series
