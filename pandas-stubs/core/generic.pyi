import sys
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Hashable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from pandas import (
    Timedelta,
    Timestamp,
)
from pandas.core.base import PandasObject
from pandas.core.indexes.base import Index
import pandas.core.indexing as indexing
from pandas.core.internals import BlockManager
from pandas.core.resample import Resampler

from pandas._typing import (
    S1,
    ArrayLike,
    Axis,
    AxisType,
    Dtype,
    FilePathOrBuffer,
    FrameOrSeries,
    FrameOrSeriesUnion,
    IgnoreRaise,
    JSONSerializable,
    Level,
    ListLike,
    Renamer,
    Scalar,
    SeriesAxisType,
    T,
)

_bool = bool
_str = str

class NDFrame(PandasObject, indexing.IndexingMixin):
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __new__(
        cls,
        data: BlockManager,
        axes: Optional[List[Index]] = ...,
        copy: _bool = ...,
        dtype: Optional[Dtype] = ...,
        attrs: Optional[Mapping[Optional[Hashable], Any]] = ...,
        fastpath: _bool = ...,
    ) -> NDFrame: ...
    def set_flags(
        self: FrameOrSeries,
        *,
        copy: bool = ...,
        allows_duplicate_labels: Optional[bool] = ...,
    ) -> FrameOrSeries: ...
    @property
    def attrs(self) -> Dict[Optional[Hashable], Any]: ...
    @attrs.setter
    def attrs(self, value: Mapping[Optional[Hashable], Any]) -> None: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def axes(self) -> List[Index]: ...
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
        float_format: Optional[_str] = ...,
        columns: Optional[Union[_str, Sequence[_str]]] = ...,
        header: _bool = ...,
        index: _bool = ...,
        index_label: Optional[Union[_str, Sequence[_str]]] = ...,
        startrow: int = ...,
        startcol: int = ...,
        engine: Optional[_str] = ...,
        merge_cells: _bool = ...,
        encoding: Optional[_str] = ...,
        inf_rep: _str = ...,
        verbose: _bool = ...,
        freeze_panes: Optional[Tuple[int, int]] = ...,
    ) -> None: ...
    @overload
    def to_json(
        self,
        path_or_buf: Optional[FilePathOrBuffer],
        orient: Optional[
            Union[
                _str, Literal["split", "records", "index", "columns", "values", "table"]
            ]
        ] = ...,
        date_format: Optional[Union[_str, Literal["epoch", "iso"]]] = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: Union[_str, Literal["s", "ms", "us", "ns"]] = ...,
        default_handler: Optional[
            Callable[[Any], Union[_str, int, float, _bool, List, Dict]]
        ] = ...,
        lines: _bool = ...,
        compression: Union[_str, Literal["infer", "gzip", "bz2", "zip", "xz"]] = ...,
        index: _bool = ...,
        indent: Optional[int] = ...,
    ) -> None: ...
    @overload
    def to_json(
        self,
        orient: Optional[
            Union[
                _str, Literal["split", "records", "index", "columns", "values", "table"]
            ]
        ] = ...,
        date_format: Optional[Union[_str, Literal["epoch", "iso"]]] = ...,
        double_precision: int = ...,
        force_ascii: _bool = ...,
        date_unit: Union[_str, Literal["s", "ms", "us", "ns"]] = ...,
        default_handler: Optional[
            Callable[[Any], Union[_str, int, float, _bool, List, Dict]]
        ] = ...,
        lines: _bool = ...,
        compression: Optional[
            Union[_str, Literal["infer", "gzip", "bz2", "zip", "xz"]]
        ] = ...,
        index: _bool = ...,
        indent: Optional[int] = ...,
    ) -> _str: ...
    def to_hdf(
        self,
        path_or_buf: FilePathOrBuffer,
        key: _str,
        mode: _str = ...,
        complevel: Optional[int] = ...,
        complib: Optional[_str] = ...,
        append: _bool = ...,
        format: Optional[_str] = ...,
        index: _bool = ...,
        min_itemsize: Optional[Union[int, Dict[_str, int]]] = ...,
        nan_rep=...,
        dropna: Optional[_bool] = ...,
        data_columns: Optional[List[_str]] = ...,
        errors: _str = ...,
        encoding: _str = ...,
    ) -> None: ...
    def to_sql(
        self,
        name: _str,
        con,
        schema: Optional[_str] = ...,
        if_exists: _str = ...,
        index: _bool = ...,
        index_label: Optional[Union[_str, Sequence[_str]]] = ...,
        chunksize: Optional[int] = ...,
        dtype: Optional[Union[Dict, Scalar]] = ...,
        method: Optional[Union[_str, Callable]] = ...,
    ) -> None: ...
    def to_pickle(
        self,
        path: _str,
        compression: Union[_str, Literal["infer", "gzip", "bz2", "zip", "xz"]] = ...,
        protocol: int = ...,
    ) -> None: ...
    def to_clipboard(
        self, excel: _bool = ..., sep: Optional[_str] = ..., **kwargs
    ) -> None: ...
    def to_xarray(self): ...
    @overload
    def to_latex(
        self,
        buf: Optional[FilePathOrBuffer],
        columns: Optional[List[_str]] = ...,
        col_space: Optional[int] = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: Optional[_bool] = ...,
        index_names: _bool = ...,
        bold_rows: _bool = ...,
        column_format: Optional[_str] = ...,
        longtable: Optional[_bool] = ...,
        escape: Optional[_bool] = ...,
        encoding: Optional[_str] = ...,
        decimal: _str = ...,
        multicolumn: Optional[_bool] = ...,
        multicolumn_format: Optional[_str] = ...,
        multirow: Optional[_bool] = ...,
        caption: Optional[Union[_str, Tuple[_str, _str]]] = ...,
        label: Optional[_str] = ...,
        position: Optional[_str] = ...,
    ) -> None: ...
    @overload
    def to_latex(
        self,
        columns: Optional[List[_str]] = ...,
        col_space: Optional[int] = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: Optional[_bool] = ...,
        index_names: _bool = ...,
        bold_rows: _bool = ...,
        column_format: Optional[_str] = ...,
        longtable: Optional[_bool] = ...,
        escape: Optional[_bool] = ...,
        encoding: Optional[_str] = ...,
        decimal: _str = ...,
        multicolumn: Optional[_bool] = ...,
        multicolumn_format: Optional[_str] = ...,
        multirow: Optional[_bool] = ...,
        caption: Optional[Union[_str, Tuple[_str, _str]]] = ...,
        label: Optional[_str] = ...,
        position: Optional[_str] = ...,
    ) -> _str: ...
    @overload
    def to_csv(
        self,
        path_or_buf: Optional[FilePathOrBuffer],
        sep: _str = ...,
        na_rep: _str = ...,
        float_format: Optional[_str] = ...,
        columns: Optional[Sequence[Hashable]] = ...,
        header: Union[_bool, List[_str]] = ...,
        index: _bool = ...,
        index_label: Optional[Union[_bool, _str, Sequence[Hashable]]] = ...,
        mode: _str = ...,
        encoding: Optional[_str] = ...,
        compression: Union[_str, Mapping[_str, _str]] = ...,
        quoting: Optional[int] = ...,
        quotechar: _str = ...,
        line_terminator: Optional[_str] = ...,
        chunksize: Optional[int] = ...,
        date_format: Optional[_str] = ...,
        doublequote: _bool = ...,
        escapechar: Optional[_str] = ...,
        decimal: _str = ...,
        errors: _str = ...,
        storage_options: Optional[Dict[_str, Any]] = ...,
    ) -> None: ...
    @overload
    def to_csv(
        self,
        sep: _str = ...,
        na_rep: _str = ...,
        float_format: Optional[_str] = ...,
        columns: Optional[Sequence[Hashable]] = ...,
        header: Union[_bool, List[_str]] = ...,
        index: _bool = ...,
        index_label: Optional[Union[_bool, _str, Sequence[Hashable]]] = ...,
        mode: _str = ...,
        encoding: Optional[_str] = ...,
        compression: Union[_str, Mapping[_str, _str]] = ...,
        quoting: Optional[int] = ...,
        quotechar: _str = ...,
        line_terminator: Optional[_str] = ...,
        chunksize: Optional[int] = ...,
        date_format: Optional[_str] = ...,
        doublequote: _bool = ...,
        escapechar: Optional[_str] = ...,
        decimal: _str = ...,
        errors: _str = ...,
        storage_options: Optional[Dict[_str, Any]] = ...,
    ) -> _str: ...
    def take(
        self, indices, axis=..., is_copy: Optional[_bool] = ..., **kwargs
    ) -> NDFrame: ...
    def xs(
        self,
        key: Union[_str, Tuple[_str]],
        axis: SeriesAxisType = ...,
        level: Optional[Level] = ...,
        drop_level: _bool = ...,
    ) -> FrameOrSeriesUnion: ...
    def __delitem__(self, idx: Hashable): ...
    def get(self, key: object, default: Optional[Dtype] = ...) -> Dtype: ...
    def reindex_like(
        self,
        other,
        method: Optional[_str] = ...,
        copy: _bool = ...,
        limit=...,
        tolerance=...,
    ) -> NDFrame: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[Hashable] = ...,
        columns: Hashable | list[Hashable] = ...,
        level: Level | None = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self: NDFrame,
        labels: Hashable | list[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[Hashable] = ...,
        columns: Hashable | list[Hashable] = ...,
        level: Level | None = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> NDFrame: ...
    @overload
    def drop(
        self: NDFrame,
        labels: Hashable | list[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[Hashable] = ...,
        columns: Hashable | list[Hashable] = ...,
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
        like: Optional[_str] = ...,
        regex: Optional[_str] = ...,
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
    ) -> Optional[NDFrame]: ...
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
        how: Optional[_str] = ...,
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
        numeric_only: Optional[_bool] = ...,
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
        datetime_is_numeric: Optional[_bool] = ...,
    ) -> NDFrame: ...
    def pct_change(
        self, periods=..., fill_method=..., limit=..., freq=..., **kwargs
    ) -> NDFrame: ...
    def transform(self, func, *args, **kwargs): ...
    def first_valid_index(self): ...
    def last_valid_index(self): ...

from pandas.core.series import Series
