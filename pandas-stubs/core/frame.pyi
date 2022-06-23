from __future__ import annotations
import datetime
import numpy as np

from pandas.core.indexing import _iLocIndexer, _LocIndexer
from matplotlib.axes import Axes as PlotAxes
from pandas._typing import (
    Axes as Axes,
    Axis as Axis,
    FilePathOrBuffer as FilePathOrBuffer,
    FilePathOrBytesBuffer as FilePathOrBytesBuffer,
    Level as Level,
    Renamer as Renamer,
)
from pandas._typing import (
    num,
    SeriesAxisType,
    AxisType,
    Dtype,
    DtypeNp,
    Label,
    StrLike,
    Scalar as Scalar,
    IndexType,
    MaskType,
    S1,
    T as TType,
)
from pandas._typing import (
    ArrayLike as ArrayLike,
    np_ndarray_str,
    np_ndarray_bool,
    Timestamp as Timestamp,
    Timedelta as Timedelta,
)
from pandas._typing import IndexLabel as IndexLabel, IgnoreRaise as IgnoreRaise
from pandas.core.arraylike import OpsMixin
from pandas.core.generic import NDFrame as NDFrame
from pandas.core.groupby.generic import DataFrameGroupBy as DataFrameGroupBy
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.base import Index as Index
from pandas.core.indexes.multi import MultiIndex as MultiIndex
from pandas.core.resample import Resampler
from pandas.core.window.rolling import Rolling, Window
from pandas.core.series import Series as Series
from pandas.io.formats import console as console, format as fmt
from pandas.io.formats.style import Styler as Styler
from pandas.plotting import PlotAccessor
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    overload,
    Pattern,
)

import numpy as _np
import datetime as _dt

_str = str
_bool = bool

class _iLocIndexerFrame(_iLocIndexer):
    @overload
    def __getitem__(self, idx: Tuple[int, int]) -> Scalar: ...
    @overload
    def __getitem__(self, idx: int) -> Series: ...
    @overload
    def __getitem__(self, idx: Tuple[Union[IndexType, MaskType], int]) -> Series: ...
    @overload
    def __getitem__(self, idx: Tuple[int, Union[IndexType, MaskType]]) -> Series: ...
    @overload
    def __getitem__(
        self,
        idx: Union[
            IndexType,
            MaskType,
            Tuple[Union[IndexType, MaskType], Union[IndexType, MaskType]],
        ],
    ) -> DataFrame: ...
    def __setitem__(
        self,
        idx: Union[
            int,
            IndexType,
            Tuple[int, int],
            Tuple[IndexType, int],
            Tuple[IndexType, IndexType],
            Tuple[int, IndexType],
        ],
        value: Union[float, Series, DataFrame],
    ) -> None: ...

class _LocIndexerFrame(_LocIndexer):
    @overload
    def __getitem__(
        self,
        idx: Union[
            Union[IndexType, MaskType],
            List[StrLike],
            Tuple[
                Union[
                    IndexType,
                    MaskType,
                    slice,
                    List[StrLike],
                    Tuple[Union[str, int, slice], ...],
                ],
                Union[List[StrLike], slice, Series[bool], Callable],
            ],
        ],
    ) -> DataFrame: ...
    @overload
    def __getitem__(
        self,
        idx: Tuple[Union[Union[int, StrLike], Tuple[StrLike, ...]], StrLike],
    ) -> Scalar: ...
    @overload
    def __getitem__(
        self,
        idx: Union[int, StrLike],
    ) -> Series: ...
    @overload
    def __getitem__(
        self,
        idx: Tuple[Union[IndexType, MaskType], StrLike],
    ) -> Series: ...
    @overload
    def __getitem__(
        self,
        idx: Tuple[Tuple[slice, ...], StrLike],
    ) -> Series: ...
    @overload
    def __setitem__(
        self,
        idx: Union[
            MaskType,
            StrLike,
            Tuple[Union[MaskType, Index, Sequence[Scalar], Scalar, slice], ...],
        ],
        value: Union[S1, ArrayLike, Series, DataFrame],
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: Tuple[Tuple[Union[StrLike, Scalar, slice], ...], StrLike],
        value: Union[S1, ArrayLike, Series[S1], List],
    ) -> None: ...
    @overload
    def __setitem__(
        self,
        idx: Tuple[Tuple[Union[StrLike, Scalar, slice], ...], StrLike],
        value: Union[S1, ArrayLike, Series[S1], List],
    ) -> None: ...

class DataFrame(NDFrame, OpsMixin):
    _ListLike = Union[
        np.ndarray,
        List[Dtype],
        Dict[_str, _np.ndarray],
        Sequence,
        Index,
        Series,
    ]
    def __new__(
        cls,
        data: Optional[Union[_ListLike, DataFrame, Dict[Any, Any]]] = ...,
        index: Optional[Axes] = ...,
        columns: Optional[Axes] = ...,
        dtype=...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    @property
    def axes(self) -> List[Index]: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def style(self) -> Styler: ...
    def items(self) -> Iterable[Tuple[Hashable, Series]]: ...
    def iteritems(self) -> Iterable[Tuple[Label, Series]]: ...
    def iterrows(self) -> Iterable[Tuple[Label, Series]]: ...
    def itertuples(self, index: _bool = ..., name: Optional[str] = ...): ...
    def __len__(self) -> int: ...
    @overload
    def dot(self, other: Union[DataFrame, ArrayLike]) -> DataFrame: ...
    @overload
    def dot(self, other: Series) -> Series: ...
    def __matmul__(self, other): ...
    def __rmatmul__(self, other): ...
    @classmethod
    def from_dict(cls, data, orient=..., dtype=..., columns=...) -> DataFrame: ...
    def to_numpy(
        self,
        dtype: Optional[Union[Type[DtypeNp], Dtype]] = ...,
        copy: _bool = ...,
        na_value: Optional[Any] = ...,
    ) -> _np.ndarray: ...
    @overload
    def to_dict(
        self,
        orient: Literal["records"],
        into: Hashable = ...,
    ) -> List[Dict[_str, Any]]: ...
    @overload
    def to_dict(
        self,
        orient: Literal["dict", "list", "series", "split", "index"] = ...,
        into: Hashable = ...,
    ) -> Dict[_str, Any]: ...
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
        columnDTypes: Optional[Union[_str, Dict]] = ...,
        indexDTypes: Optional[Union[_str, Dict]] = ...,
    ) -> np.recarray: ...
    def to_stata(
        self,
        path: FilePathOrBuffer,
        convert_dates: Optional[Dict] = ...,
        write_index: _bool = ...,
        byteorder: Optional[Union[_str, Literal["<", ">", "little", "big"]]] = ...,
        time_stamp=...,
        data_label: Optional[_str] = ...,
        variable_labels: Optional[Dict] = ...,
        version: int = ...,
        convert_strl: Optional[List[_str]] = ...,
    ) -> None: ...
    def to_feather(self, path: FilePathOrBuffer, **kwargs) -> None: ...
    @overload
    def to_markdown(
        self, buf: Optional[FilePathOrBuffer], mode: Optional[_str] = ..., **kwargs
    ) -> None: ...
    @overload
    def to_markdown(self, mode: Optional[_str] = ..., **kwargs) -> _str: ...
    @overload
    def to_parquet(
        self,
        path: FilePathOrBytesBuffer,
        *,
        engine: Union[_str, Literal["auto", "pyarrow", "fastparquet"]] = ...,
        compression: Union[_str, Literal["snappy", "gzip", "brotli"]] = ...,
        index: Optional[_bool] = ...,
        partition_cols: Optional[List] = ...,
        **kwargs,
    ) -> None: ...
    @overload
    def to_parquet(
        self,
        *,
        path: None = ...,
        engine: Union[_str, Literal["auto", "pyarrow", "fastparquet"]] = ...,
        compression: Union[_str, Literal["snappy", "gzip", "brotli"]] = ...,
        index: Optional[_bool] = ...,
        partition_cols: Optional[List] = ...,
        **kwargs,
    ) -> bytes: ...
    @overload
    def to_html(
        self,
        buf: Optional[FilePathOrBuffer],
        columns: Optional[Sequence[_str]] = ...,
        col_space: Optional[Union[int, List[int], Dict[Union[_str, int], int]]] = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: Optional[_bool] = ...,
        index_names: _bool = ...,
        justify: Optional[_str] = ...,
        max_rows: Optional[int] = ...,
        max_cols: Optional[int] = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        bold_rows: _bool = ...,
        classes: Optional[Union[_str, List, Tuple]] = ...,
        escape: _bool = ...,
        notebook: _bool = ...,
        border: Optional[int] = ...,
        table_id: Optional[_str] = ...,
        render_links: _bool = ...,
        encoding: Optional[_str] = ...,
    ) -> None: ...
    @overload
    def to_html(
        self,
        columns: Optional[Sequence[_str]] = ...,
        col_space: Optional[Union[int, List[int], Dict[Union[_str, int], int]]] = ...,
        header: _bool = ...,
        index: _bool = ...,
        na_rep: _str = ...,
        formatters=...,
        float_format=...,
        sparsify: Optional[_bool] = ...,
        index_names: _bool = ...,
        justify: Optional[_str] = ...,
        max_rows: Optional[int] = ...,
        max_cols: Optional[int] = ...,
        show_dimensions: _bool = ...,
        decimal: _str = ...,
        bold_rows: _bool = ...,
        classes: Optional[Union[_str, List, Tuple]] = ...,
        escape: _bool = ...,
        notebook: _bool = ...,
        border: Optional[int] = ...,
        table_id: Optional[_str] = ...,
        render_links: _bool = ...,
        encoding: Optional[_str] = ...,
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
        idx: Union[
            Tuple,
            Series[_bool],
            DataFrame,
            List[_str],
            List[Scalar],
            Index,
            np_ndarray_str,
            np_ndarray_bool,
            Sequence[Tuple[Scalar, ...]],
        ],
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
        include: Optional[Union[_str, List[_str]]] = ...,
        exclude: Optional[Union[_str, List[_str]]] = ...,
    ) -> DataFrame: ...
    def insert(
        self,
        loc: int,
        column,
        value: Union[int, _ListLike],
        allow_duplicates: _bool = ...,
    ) -> None: ...
    def assign(self, **kwargs) -> DataFrame: ...
    def lookup(self, row_labels: Sequence, col_labels: Sequence) -> np.ndarray: ...
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
        fill_axis: AxisType = ...,
        broadcast_axis: Optional[AxisType] = ...,
    ) -> DataFrame: ...
    def reindex(**kwargs) -> DataFrame: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[Hashable] = ...,
        columns: Hashable | list[Hashable] = ...,
        level: Optional[Level] = ...,
        inplace: Literal[True],
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[Hashable] = ...,
        columns: Hashable | list[Hashable] = ...,
        level: Optional[Level] = ...,
        inplace: Literal[False] = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame: ...
    @overload
    def drop(
        self,
        labels: Hashable | list[Hashable] = ...,
        *,
        axis: Axis = ...,
        index: Hashable | list[Hashable] = ...,
        columns: Hashable | list[Hashable] = ...,
        level: Optional[Level] = ...,
        inplace: bool = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame | None: ...
    @overload
    def rename(
        self,
        mapper: Optional[Renamer] = ...,
        *,
        index: Optional[Renamer] = ...,
        columns: Optional[Renamer] = ...,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: Literal[True],
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> None: ...
    @overload
    def rename(
        self,
        mapper: Optional[Renamer] = ...,
        *,
        index: Optional[Renamer] = ...,
        columns: Optional[Renamer] = ...,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: Literal[False] = ...,
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> DataFrame: ...
    @overload
    def rename(
        self,
        mapper: Optional[Renamer] = ...,
        *,
        index: Optional[Renamer] = ...,
        columns: Optional[Renamer] = ...,
        axis: Optional[Axis] = ...,
        copy: bool = ...,
        inplace: bool = ...,
        level: Optional[Level] = ...,
        errors: IgnoreRaise = ...,
    ) -> Optional[DataFrame]: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series, DataFrame]] = ...,
        method: Optional[Literal["backfill", "bfill", "ffill", "pad"]] = ...,
        axis: Optional[AxisType] = ...,
        limit: int = ...,
        downcast: Optional[Dict] = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series, DataFrame]] = ...,
        method: Optional[Literal["backfill", "bfill", "ffill", "pad"]] = ...,
        axis: Optional[AxisType] = ...,
        limit: int = ...,
        downcast: Optional[Dict] = ...,
        *,
        inplace: Literal[False] = ...,
    ) -> DataFrame: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series, DataFrame]] = ...,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "ffill", "pad"]]
        ] = ...,
        axis: Optional[AxisType] = ...,
        *,
        limit: int = ...,
        downcast: Optional[Dict] = ...,
    ) -> Union[None, DataFrame]: ...
    @overload
    def fillna(
        self,
        value: Optional[Union[Scalar, Dict, Series, DataFrame]] = ...,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "ffill", "pad"]]
        ] = ...,
        axis: Optional[AxisType] = ...,
        inplace: Optional[_bool] = ...,
        limit: int = ...,
        downcast: Optional[Dict] = ...,
    ) -> Union[None, DataFrame]: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Optional[Union[Scalar, Sequence, Mapping, Pattern]] = ...,
        limit: Optional[int] = ...,
        regex=...,
        method: Optional[_str] = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Optional[Union[Scalar, Sequence, Mapping, Pattern]] = ...,
        limit: Optional[int] = ...,
        regex=...,
        method: Optional[_str] = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Optional[Union[Scalar, Sequence, Mapping, Pattern]] = ...,
        *,
        limit: Optional[int] = ...,
        regex=...,
        method: Optional[_str] = ...,
    ) -> DataFrame: ...
    @overload
    def replace(
        self,
        to_replace=...,
        value: Optional[Union[Scalar, Sequence, Mapping, Pattern]] = ...,
        inplace: Optional[_bool] = ...,
        limit: Optional[int] = ...,
        regex=...,
        method: Optional[_str] = ...,
    ) -> Union[None, DataFrame]: ...
    def shift(
        self,
        periods: int = ...,
        freq=...,
        axis: AxisType = ...,
        fill_value: Optional[Hashable] = ...,
    ) -> DataFrame: ...
    @overload
    def set_index(
        self,
        keys: Union[Label, Sequence],
        drop: _bool = ...,
        append: _bool = ...,
        verify_integrity: _bool = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def set_index(
        self,
        keys: Union[Label, Sequence],
        drop: _bool = ...,
        append: _bool = ...,
        verify_integrity: _bool = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def set_index(
        self,
        keys: Union[Label, Sequence],
        drop: _bool = ...,
        append: _bool = ...,
        *,
        verify_integrity: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def set_index(
        self,
        keys: Union[Label, Sequence],
        drop: _bool = ...,
        append: _bool = ...,
        inplace: Optional[_bool] = ...,
        verify_integrity: _bool = ...,
    ) -> Union[None, DataFrame]: ...
    @overload
    def reset_index(
        self,
        level: Union[Level, Sequence[Level]] = ...,
        drop: _bool = ...,
        col_level: Union[int, _str] = ...,
        col_fill: Hashable = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def reset_index(
        self,
        level: Union[Level, Sequence[Level]] = ...,
        drop: _bool = ...,
        col_level: Union[int, _str] = ...,
        col_fill: Hashable = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Union[Level, Sequence[Level]] = ...,
        drop: _bool = ...,
        *,
        col_level: Union[int, _str] = ...,
        col_fill: Hashable = ...,
    ) -> DataFrame: ...
    @overload
    def reset_index(
        self,
        level: Union[Level, Sequence[Level]] = ...,
        drop: _bool = ...,
        inplace: Optional[_bool] = ...,
        col_level: Union[int, _str] = ...,
        col_fill: Hashable = ...,
    ) -> Union[None, DataFrame]: ...
    def isna(self) -> DataFrame: ...
    def isnull(self) -> DataFrame: ...
    def notna(self) -> DataFrame: ...
    def notnull(self) -> DataFrame: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: Union[_str, Literal["any", "all"]] = ...,
        thresh: Optional[int] = ...,
        subset: Optional[List] = ...,
        *,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: Union[_str, Literal["any", "all"]] = ...,
        thresh: Optional[int] = ...,
        subset: Optional[List] = ...,
        *,
        inplace: Literal[False],
    ) -> DataFrame: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: Union[_str, Literal["any", "all"]] = ...,
        thresh: Optional[int] = ...,
        subset: Optional[List] = ...,
    ) -> DataFrame: ...
    @overload
    def dropna(
        self,
        axis: AxisType = ...,
        how: Union[_str, Literal["any", "all"]] = ...,
        thresh: Optional[int] = ...,
        subset: Optional[List] = ...,
        inplace: Optional[_bool] = ...,
    ) -> Union[None, DataFrame]: ...
    def drop_duplicates(
        self,
        subset=...,
        keep: Union[_str, Literal["first", "last"], _bool] = ...,
        inplace: _bool = ...,
        ignore_index: _bool = ...,
    ) -> DataFrame: ...
    def duplicated(
        self,
        subset: Optional[Union[Hashable, Sequence[Hashable]]] = ...,
        keep: Union[_str, Literal["first", "last"], _bool] = ...,
    ) -> Series: ...
    @overload
    def sort_values(
        self,
        by: Union[_str, Sequence[_str]],
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
        by: Union[_str, Sequence[_str]],
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        *,
        inplace: Literal[False],
        key: Optional[Callable] = ...,
    ) -> DataFrame: ...
    @overload
    def sort_values(
        self,
        by: Union[_str, Sequence[_str]],
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        *,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        key: Optional[Callable] = ...,
    ) -> DataFrame: ...
    @overload
    def sort_values(
        self,
        by: Union[_str, Sequence[_str]],
        axis: AxisType = ...,
        ascending: Union[_bool, Sequence[_bool]] = ...,
        inplace: Optional[_bool] = ...,
        kind: Union[_str, Literal["quicksort", "mergesort", "heapsort"]] = ...,
        na_position: Union[_str, Literal["first", "last"]] = ...,
        ignore_index: _bool = ...,
        key: Optional[Callable] = ...,
    ) -> Union[None, DataFrame]: ...
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
    ) -> DataFrame: ...
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
    ) -> DataFrame: ...
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
    ) -> Union[None, DataFrame]: ...
    def value_counts(
        self,
        subset: Optional[Sequence[Hashable]] = ...,
        normalize: _bool = ...,
        sort: _bool = ...,
        ascending: _bool = ...,
        dropna: _bool = ...,
    ) -> Series[int]: ...
    def nlargest(
        self,
        n: int,
        columns: Union[_str, List[_str]],
        keep: Union[_str, Literal["first", "last", "all"]] = ...,
    ) -> DataFrame: ...
    def nsmallest(
        self,
        n: int,
        columns: Union[_str, List[_str]],
        keep: Union[_str, Literal["first", "last", "all"]] = ...,
    ) -> DataFrame: ...
    def swaplevel(
        self, i: Level = ..., j: Level = ..., axis: AxisType = ...
    ) -> DataFrame: ...
    def reorder_levels(self, order: List, axis: AxisType = ...) -> DataFrame: ...
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
        other: Union[DataFrame, Series],
        join: _str = ...,
        overwrite: _bool = ...,
        filter_func: Optional[Callable] = ...,
        errors: Union[_str, Literal["raise", "ignore"]] = ...,
    ) -> None: ...
    def groupby(
        self,
        by: Optional[Union[List[_str], _str]] = ...,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        as_index: _bool = ...,
        sort: _bool = ...,
        group_keys: _bool = ...,
        squeeze: _bool = ...,
        observed: _bool = ...,
        dropna: _bool = ...,
    ) -> DataFrameGroupBy: ...
    def pivot(
        self,
        index=...,
        columns=...,
        values=...,
    ) -> DataFrame: ...
    def pivot_table(
        self,
        values: Optional[_str] = ...,
        index: Optional[Union[_str, Grouper, Sequence]] = ...,
        columns: Optional[Union[_str, Grouper, Sequence]] = ...,
        aggfunc=...,
        fill_value: Optional[Scalar] = ...,
        margins: _bool = ...,
        dropna: _bool = ...,
        margins_name: _str = ...,
        observed: _bool = ...,
    ) -> DataFrame: ...
    def stack(
        self, level: Level = ..., dropna: _bool = ...
    ) -> Union[DataFrame, Series]: ...
    def explode(
        self, column: Union[_str, Tuple], ignore_index: _bool = ...
    ) -> DataFrame: ...
    def unstack(
        self,
        level: Level = ...,
        fill_value: Optional[Union[int, _str, Dict]] = ...,
    ) -> Union[DataFrame, Series]: ...
    def melt(
        self,
        id_vars: Optional[Union[Tuple, Sequence, np.ndarray]] = ...,
        value_vars: Optional[Union[Tuple, Sequence, np.ndarray]] = ...,
        var_name: Optional[Scalar] = ...,
        value_name: Scalar = ...,
        col_level: Optional[Union[int, _str]] = ...,
        ignore_index: _bool = ...,
    ) -> DataFrame: ...
    def diff(self, periods: int = ..., axis: AxisType = ...) -> DataFrame: ...
    @overload
    def agg(
        self, func: Union[Callable, _str], axis: AxisType = ..., **kwargs
    ) -> Series: ...
    @overload
    def agg(
        self,
        func: Union[List[Callable], Dict[_str, Callable]] = ...,
        axis: AxisType = ...,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def aggregate(
        self, func: Union[Callable, _str], axis: AxisType = ..., **kwargs
    ) -> Series: ...
    @overload
    def aggregate(
        self,
        func: Union[List[Callable], Dict[_str, Callable]],
        axis: AxisType = ...,
        **kwargs,
    ) -> DataFrame: ...
    def transform(
        self,
        func: Union[List[Callable], Dict[_str, Callable]],
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
        result_type: Optional[_str] = ...,
        args=...,
        **kwargs,
    ) -> DataFrame: ...
    def applymap(
        self, func: Callable, na_action: Optional[Literal["ignore"]] = None, **kwargs
    ) -> DataFrame: ...
    def append(
        self,
        other: Union[
            DataFrame, Series, Dict[Any, Any], Sequence[Scalar], Sequence[_ListLike]
        ],
        ignore_index: _bool = ...,
        verify_integrity: _bool = ...,
        sort: _bool = ...,
    ) -> DataFrame: ...
    def join(
        self,
        other: Union[DataFrame, Series, List[Union[DataFrame, Series]]],
        on: Optional[Union[_str, List[_str]]] = ...,
        how: Union[_str, Literal["left", "right", "outer", "inner"]] = ...,
        lsuffix: _str = ...,
        rsuffix: _str = ...,
        sort: _bool = ...,
    ) -> DataFrame: ...
    def merge(
        self,
        right: Union[DataFrame, Series],
        how: Union[_str, Literal["left", "right", "inner", "outer"]] = ...,
        on: Optional[IndexLabel] = ...,
        left_on: Optional[IndexLabel] = ...,
        right_on: Optional[IndexLabel] = ...,
        left_index: _bool = ...,
        right_index: _bool = ...,
        sort: _bool = ...,
        suffixes: Tuple[Optional[_str], Optional[_str]] = ...,
        copy: _bool = ...,
        indicator: Union[_bool, _str] = ...,
        validate: Optional[_str] = ...,
    ) -> DataFrame: ...
    def round(
        self, decimals: Union[int, Dict, Series] = ..., *args, **kwargs
    ) -> DataFrame: ...
    def corr(
        self,
        method: Union[_str, Literal["pearson", "kendall", "spearman"]] = ...,
        min_periods: int = ...,
    ) -> DataFrame: ...
    def cov(self, min_periods: Optional[int] = ..., ddof: int = 1) -> DataFrame: ...
    def corrwith(
        self,
        other: Union[DataFrame, Series],
        axis: Optional[AxisType] = ...,
        drop: _bool = ...,
        method: Union[_str, Literal["pearson", "kendall", "spearman"]] = ...,
    ) -> Series: ...
    @overload
    def count(
        self, axis: AxisType = ..., numeric_only: _bool = ..., *, level: Level
    ) -> DataFrame: ...
    @overload
    def count(
        self, axis: AxisType = ..., level: None = ..., numeric_only: _bool = ...
    ) -> Series: ...
    def nunique(self, axis: AxisType = ..., dropna=True) -> Series: ...
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
        interpolation: Union[
            _str, Literal["linear", "lower", "higher", "midpoint", "nearest"]
        ] = ...,
    ) -> Series: ...
    @overload
    def quantile(
        self,
        q: List[float],
        axis: AxisType = ...,
        numeric_only: _bool = ...,
        interpolation: Union[
            _str, Literal["linear", "lower", "higher", "midpoint", "nearest"]
        ] = ...,
    ) -> DataFrame: ...
    def to_timestamp(
        self,
        freq=...,
        how: Union[_str, Literal["start", "end", "s", "e"]] = ...,
        axis: AxisType = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    def to_period(
        self, freq: Optional[_str] = ..., axis: AxisType = ..., copy: _bool = ...
    ) -> DataFrame: ...
    def isin(self, values: Union[Iterable, Series, DataFrame, Dict]) -> DataFrame: ...
    def plot(self, *args, **kwargs) -> PlotAccessor: ...
    def hist(
        self,
        column: Optional[Union[_str, List[_str]]] = ...,
        by: Optional[Union[_str, _ListLike]] = ...,
        grid: _bool = ...,
        xlabelsize: Optional[int] = ...,
        xrot: Optional[float] = ...,
        ylabelsize: Optional[int] = ...,
        yrot: Optional[float] = ...,
        ax: Optional[PlotAxes] = ...,
        sharex: _bool = ...,
        sharey: _bool = ...,
        figsize: Optional[Tuple[float, float]] = ...,
        layout: Optional[Tuple[int, int]] = ...,
        bins: Union[int, List] = ...,
        backend: Optional[_str] = ...,
        **kwargs,
    ): ...
    def boxplot(
        self,
        column: Optional[Union[_str, List[_str]]] = ...,
        by: Optional[Union[_str, _ListLike]] = ...,
        ax: Optional[PlotAxes] = ...,
        fontsize: Optional[Union[float, _str]] = ...,
        rot: int = ...,
        grid: _bool = ...,
        figsize: Optional[Tuple[float, float]] = ...,
        layout: Optional[Tuple[int, int]] = ...,
        return_type: Optional[Union[_str, Literal["axes", "dict", "both"]]] = ...,
        backend: Optional[_str] = ...,
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
        other: Union[num, _ListLike, DataFrame],
        axis: AxisType = ...,
        level: Level = ...,
        fill_value: Union[None, float] = ...,
    ) -> DataFrame: ...
    def __iter__(self) -> Iterator[Hashable]: ...
    # properties
    @property
    def at(self): ...  # Not sure what to do with this yet; look at source
    @property
    def bool(self) -> _bool: ...
    @property
    def columns(self) -> Index: ...
    @columns.setter  # setter needs to be right next to getter; otherwise mypy complains
    def columns(self, cols: Union[List[_str], Index[_str]]) -> None: ...  # type:ignore
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
    def values(self) -> _np.ndarray: ...
    # methods
    def abs(self) -> DataFrame: ...
    def add(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def add_prefix(self, prefix: _str) -> DataFrame: ...
    def add_suffix(self, suffix: _str) -> DataFrame: ...
    @overload
    def all(
        self,
        axis: AxisType = ...,
        bool_only: Optional[_bool] = ...,
        skipna: _bool = ...,
        level: None = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def all(
        self,
        axis: AxisType = ...,
        bool_only: Optional[_bool] = ...,
        skipna: _bool = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def any(
        self,
        axis: AxisType = ...,
        bool_only: Optional[_bool] = ...,
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
    def asof(
        self, where, subset: Optional[Union[_str, List[_str]]] = ...
    ) -> DataFrame: ...
    def asfreq(
        self,
        freq,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill"]]
        ] = ...,
        how: Optional[Union[_str, Literal["start", "end"]]] = ...,
        normalize: _bool = ...,
        fill_value: Optional[Scalar] = ...,
    ) -> DataFrame: ...
    def astype(
        self,
        dtype: Union[_str, Dtype, Dict[_str, Union[_str, Dtype]]],
        copy: _bool = ...,
        errors: _str = ...,
    ) -> DataFrame: ...
    def at_time(
        self,
        time: Union[_str, datetime.time],
        asof: _bool = ...,
        axis: Optional[AxisType] = ...,
    ) -> DataFrame: ...
    def between_time(
        self,
        start_time: Union[_str, datetime.time],
        end_time: Union[_str, datetime.time],
        include_start: _bool = ...,
        include_end: _bool = ...,
        axis: Optional[AxisType] = ...,
    ) -> DataFrame: ...
    @overload
    def bfill(
        self,
        axis: Optional[AxisType] = ...,
        *,
        inplace: Literal[True],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> None: ...
    @overload
    def bfill(
        self,
        axis: Optional[AxisType] = ...,
        *,
        inplace: Literal[False],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> DataFrame: ...
    def clip(
        self,
        lower: Optional[float] = ...,
        upper: Optional[float] = ...,
        axis: Optional[AxisType] = ...,
        inplace: _bool = ...,
        *args,
        **kwargs,
    ) -> DataFrame: ...
    def copy(self, deep: _bool = ...) -> DataFrame: ...
    def cummax(
        self, axis: Optional[AxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def cummin(
        self, axis: Optional[AxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def cumprod(
        self, axis: Optional[AxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def cumsum(
        self, axis: Optional[AxisType] = ..., skipna: _bool = ..., *args, **kwargs
    ) -> DataFrame: ...
    def describe(
        self,
        percentiles: Optional[List[float]] = ...,
        include: Optional[Union[_str, Literal["all"], List[Dtype]]] = ...,
        exclude: Optional[List[Dtype]] = ...,
        datetime_is_numeric: Optional[_bool] = ...,
    ) -> DataFrame: ...
    def div(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def divide(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def droplevel(
        self, level: Union[Level, List[Level]] = ..., axis: AxisType = ...
    ) -> DataFrame: ...
    def eq(
        self, other, axis: AxisType = ..., level: Optional[Level] = ...
    ) -> DataFrame: ...
    def equals(self, other: Union[Series, DataFrame]) -> _bool: ...
    def ewm(
        self,
        com: Optional[float] = ...,
        span: Optional[float] = ...,
        halflife: Optional[float] = ...,
        alpha: Optional[float] = ...,
        min_periods: int = ...,
        adjust: _bool = ...,
        ignore_na: _bool = ...,
        axis: AxisType = ...,
    ) -> DataFrame: ...
    def exp(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def expanding(
        self, min_periods: int = ..., center: _bool = ..., axis: AxisType = ...
    ): ...  # for now
    @overload
    def ffill(
        self,
        axis: Optional[AxisType] = ...,
        *,
        inplace: Literal[True],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> None: ...
    @overload
    def ffill(
        self,
        axis: Optional[AxisType] = ...,
        *,
        inplace: Literal[False],
        limit: Optional[int] = ...,
        downcast: Optional[Dict] = ...,
    ) -> DataFrame: ...
    def filter(
        self,
        items: Optional[List] = ...,
        like: Optional[_str] = ...,
        regex: Optional[_str] = ...,
        axis: Optional[AxisType] = ...,
    ) -> DataFrame: ...
    def first(self, offset) -> DataFrame: ...
    def first_valid_index(self) -> Scalar: ...
    def floordiv(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    # def from_dict
    # def from_records
    def fulldiv(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def ge(
        self, other, axis: AxisType = ..., level: Optional[Level] = ...
    ) -> DataFrame: ...
    # def get
    def gt(
        self, other, axis: AxisType = ..., level: Optional[Level] = ...
    ) -> DataFrame: ...
    def head(self, n: int = ...) -> DataFrame: ...
    def infer_objects(self) -> DataFrame: ...
    # def info
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: Optional[int] = ...,
        limit_direction: Union[_str, Literal["forward", "backward", "both"]] = ...,
        limit_area: Union[_str, Optional[Literal["inside", "outside"]]] = ...,
        downcast: Optional[Union[_str, Literal["infer"]]] = ...,
        *,
        inplace: Literal[True],
        **kwargs,
    ) -> None: ...
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: Optional[int] = ...,
        limit_direction: Union[_str, Literal["forward", "backward", "both"]] = ...,
        limit_area: Union[_str, Optional[Literal["inside", "outside"]]] = ...,
        downcast: Optional[Union[_str, Literal["infer"]]] = ...,
        *,
        inplace: Literal[False],
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: Optional[int] = ...,
        limit_direction: Union[_str, Literal["forward", "backward", "both"]] = ...,
        limit_area: Union[_str, Optional[Literal["inside", "outside"]]] = ...,
        downcast: Optional[Union[_str, Literal["infer"]]] = ...,
    ) -> DataFrame: ...
    @overload
    def interpolate(
        self,
        method: _str = ...,
        axis: AxisType = ...,
        limit: Optional[int] = ...,
        inplace: Optional[_bool] = ...,
        limit_direction: Union[_str, Literal["forward", "backward", "both"]] = ...,
        limit_area: Optional[Union[_str, Literal["inside", "outside"]]] = ...,
        downcast: Optional[Union[_str, Literal["infer"]]] = ...,
        **kwargs,
    ) -> DataFrame: ...
    def keys(self) -> Index: ...
    @overload
    def kurt(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def kurt(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def kurtosis(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def kurtosis(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    def last(self, offset) -> DataFrame: ...
    def last_valid_index(self) -> Scalar: ...
    def le(
        self, other, axis: AxisType = ..., level: Optional[Level] = ...
    ) -> DataFrame: ...
    def lt(
        self, other, axis: AxisType = ..., level: Optional[Level] = ...
    ) -> DataFrame: ...
    @overload
    def mad(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
    ) -> Series: ...
    @overload
    def mad(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    def mask(
        self,
        cond: Union[Series, DataFrame, _np.ndarray],
        other=...,
        inplace: _bool = ...,
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def max(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def max(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def mean(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def mean(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def median(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def median(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    @overload
    def min(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def min(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    def mod(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def mul(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def multiply(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def ne(
        self, other, axis: AxisType = ..., level: Optional[Level] = ...
    ) -> DataFrame: ...
    def pct_change(
        self,
        periods: int = ...,
        fill_method: _str = ...,
        limit: Optional[int] = ...,
        freq=...,
        **kwargs,
    ) -> DataFrame: ...
    def pipe(
        self,
        func: Callable[..., TType] | tuple[Callable[..., TType], str],
        *args,
        **kwargs,
    ) -> TType: ...
    def pop(self, item: _str) -> Series: ...
    def pow(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    @overload
    def prod(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def prod(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Series: ...
    def product(
        self,
        axis: Optional[AxisType] = ...,
        skipna: _bool = ...,
        level: Optional[Level] = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        **kwargs,
    ) -> DataFrame: ...
    def radd(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def rank(
        self,
        axis: AxisType = ...,
        method: Union[_str, Literal["average", "min", "max", "first", "dense"]] = ...,
        numeric_only: Optional[_bool] = ...,
        na_option: Union[_str, Literal["keep", "top", "bottom"]] = ...,
        ascending: _bool = ...,
        pct: _bool = ...,
    ) -> DataFrame: ...
    def rdiv(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def reindex_like(
        self,
        other: DataFrame,
        method: Optional[
            Union[_str, Literal["backfill", "bfill", "pad", "ffill", "nearest"]]
        ] = ...,
        copy: _bool = ...,
        limit: Optional[int] = ...,
        tolerance=...,
    ) -> DataFrame: ...
    @overload
    def rename_axis(
        self,
        mapper=...,
        *,
        inplace: Literal[True],
        axis: Optional[AxisType] = ...,
        copy: _bool = ...,
    ) -> None: ...
    @overload
    def rename_axis(
        self,
        mapper=...,
        *,
        inplace: Literal[False] = ...,
        axis: Optional[AxisType] = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    @overload
    def rename_axis(
        self,
        *,
        inplace: Literal[True],
        index: Optional[
            Union[_str, Sequence[_str], Dict[Union[_str, int], _str], Callable]
        ] = ...,
        columns: Optional[
            Union[_str, Sequence[_str], Dict[Union[_str, int], _str], Callable]
        ] = ...,
        copy: _bool = ...,
    ) -> None: ...
    @overload
    def rename_axis(
        self,
        *,
        inplace: Literal[False] = ...,
        index: Optional[
            Union[_str, Sequence[_str], Dict[Union[_str, int], _str], Callable]
        ] = ...,
        columns: Optional[
            Union[_str, Sequence[_str], Dict[Union[_str, int], _str], Callable]
        ] = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    def resample(
        self,
        rule,
        axis: AxisType = ...,
        closed: Optional[_str] = ...,
        label: Optional[_str] = ...,
        convention: Union[_str, Literal["start", "end", "s", "e"]] = ...,
        kind: Union[_str, Optional[Literal["timestamp", "period"]]] = ...,
        loffset=...,
        base: int = ...,
        on: Optional[_str] = ...,
        level: Optional[Level] = ...,
        origin: Union[
            Timestamp, Literal["epoch", "start", "start_day", "end", "end_day"]
        ] = ...,
        offset: Optional[Union[Timedelta, _str]] = None,
    ) -> Resampler: ...
    def rfloordiv(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[Union[float, None]] = ...,
    ) -> DataFrame: ...
    def rmod(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def rmul(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    @overload
    def rolling(
        self,
        window,
        min_periods: Optional[int] = ...,
        center: _bool = ...,
        *,
        win_type: _str,
        on: Optional[_str] = ...,
        axis: AxisType = ...,
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
        axis: AxisType = ...,
        closed: Optional[_str] = ...,
    ) -> Rolling: ...
    def rpow(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def rsub(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def rtruediv(
        self,
        other,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    # sample is missing a weights arg
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
    @overload
    def sem(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def sem(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
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
        inplace: Optional[_bool] = ...,
    ) -> Union[None, DataFrame]: ...
    @overload
    def skew(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def skew(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    def slice_shift(self, periods: int = ..., axis: AxisType = ...) -> DataFrame: ...
    def squeeze(self, axis: Optional[AxisType] = ...): ...
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
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def subtract(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    @overload
    def sum(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def sum(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        numeric_only: Optional[_bool] = ...,
        min_count: int = ...,
        **kwargs,
    ) -> Series: ...
    def swapaxes(
        self, axis1: AxisType, axis2: AxisType, copy: _bool = ...
    ) -> DataFrame: ...
    def tail(self, n: int = ...) -> DataFrame: ...
    def take(
        self,
        indices: List,
        axis: AxisType = ...,
        is_copy: Optional[_bool] = ...,
        **kwargs,
    ) -> DataFrame: ...
    def tshift(
        self, periods: int = ..., freq=..., axis: AxisType = ...
    ) -> DataFrame: ...
    def to_clipboard(
        self, excel: _bool = ..., sep: Optional[_str] = ..., **kwargs
    ) -> None: ...
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
        compression: Union[
            _str, None, Literal["infer", "gzip", "bz2", "zip", "xz"]
        ] = ...,
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
        compression: Union[
            _str, None, Literal["infer", "gzip", "bz2", "zip", "xz"]
        ] = ...,
        index: _bool = ...,
        indent: Optional[int] = ...,
    ) -> _str: ...
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
        position: Optional[str] = ...,
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
        position: Optional[str] = ...,
    ) -> _str: ...
    def to_pickle(
        self,
        path: _str,
        compression: Union[_str, Literal["infer", "gzip", "bz2", "zip", "xz"]] = ...,
        protocol: int = ...,
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
    @overload
    def to_string(
        self,
        buf: Optional[FilePathOrBuffer],
        columns: Optional[Sequence[_str]] = ...,
        col_space: Optional[Union[int, List[int], Dict[Union[_str, int], int]]] = ...,
        header: Union[_bool, Sequence[_str]] = ...,
        index: _bool = ...,
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
        columns: Optional[Sequence[_str]] = ...,
        col_space: Optional[Union[int, List[int], Dict[Union[_str, int], int]]] = ...,
        header: Union[_bool, Sequence[_str]] = ...,
        index: _bool = ...,
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
    def to_xarray(self): ...
    def truediv(
        self,
        other: Union[num, _ListLike, DataFrame],
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        fill_value: Optional[float] = ...,
    ) -> DataFrame: ...
    def truncate(
        self,
        before: Optional[Union[datetime.date, _str, int]] = ...,
        after: Optional[Union[datetime.date, _str, int]] = ...,
        axis: Optional[AxisType] = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    # def tshift
    def tz_convert(
        self,
        tz,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        copy: _bool = ...,
    ) -> DataFrame: ...
    def tz_localize(
        self,
        tz,
        axis: AxisType = ...,
        level: Optional[Level] = ...,
        copy: _bool = ...,
        ambiguous=...,
        nonexistent: _str = ...,
    ) -> DataFrame: ...
    def unique(self) -> DataFrame: ...
    @overload
    def var(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        *,
        level: Level,
        **kwargs,
    ) -> DataFrame: ...
    @overload
    def var(
        self,
        axis: Optional[AxisType] = ...,
        skipna: Optional[_bool] = ...,
        level: None = ...,
        ddof: int = ...,
        numeric_only: Optional[_bool] = ...,
        **kwargs,
    ) -> Series: ...
    def where(
        self,
        cond: Union[Series, DataFrame, _np.ndarray],
        other=...,
        inplace: _bool = ...,
        axis: Optional[AxisType] = ...,
        level: Optional[Level] = ...,
        errors: _str = ...,
        try_cast: _bool = ...,
    ) -> DataFrame: ...
    # Move from generic because Series is Generic and it returns Series[bool] there
    def __invert__(self) -> DataFrame: ...
