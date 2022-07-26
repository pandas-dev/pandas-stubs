from __future__ import annotations

from typing import (
    Any,
    Hashable,
    Literal,
)

from _typeshed import Incomplete
import numpy as np
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Series,
)
from pandas.core.generic import NDFrame
from tables import (
    Col,
    Node,
)

from pandas._typing import (
    AnyArrayLike,
    ArrayLike,
    DtypeArg,
    FilePathOrBuffer,
    npt,
)

# from tables import Col, File, Node
# pytables may not be installed so create them as dummy classes

class PossibleDataLossError(Exception): ...
class ClosedFileError(Exception): ...
class IncompatibilityWarning(Warning): ...
class AttributeConflictWarning(Warning): ...
class DuplicateWarning(Warning): ...

def to_hdf(
    path_or_buf: FilePathOrBuffer,
    key: str,
    value: NDFrame,
    mode: str = ...,
    complevel: int | None = ...,
    complib: str | None = ...,
    append: bool = ...,
    format: str | None = ...,
    index: bool = ...,
    min_itemsize: int | dict[str, int] | None = ...,
    nan_rep: str | None = ...,
    dropna: bool | None = ...,
    data_columns: Literal[True] | list[str] | None = ...,
    errors: str = ...,
    encoding: str = ...,
): ...
def read_hdf(
    path_or_buf: FilePathOrBuffer,
    key=...,
    mode: str = ...,
    errors: str = ...,
    where: list[Any] | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[str] | None = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    **kwargs,
): ...

class HDFStore:
    def __init__(
        self,
        path,
        mode: str = ...,
        complevel: int | None = ...,
        complib=...,
        fletcher32: bool = ...,
        **kwargs,
    ) -> None: ...
    def __fspath__(self): ...
    @property
    def root(self): ...
    @property
    def filename(self): ...
    def __getitem__(self, key: str): ...
    def __setitem__(self, key: str, value): ...
    def __delitem__(self, key: str): ...
    def __getattr__(self, name: str): ...
    def __contains__(self, key: str) -> bool: ...
    def __len__(self) -> int: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def keys(self) -> list[str]: ...
    def __iter__(self): ...
    def items(self) -> None: ...
    iteritems = ...
    def open(self, mode: str = ..., **kwargs): ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...
    def flush(self, fsync: bool = ...): ...
    def get(self, key: str): ...
    def select(
        self,
        key: str,
        where=...,
        start=...,
        stop=...,
        columns=...,
        iterator=...,
        chunksize=...,
        auto_close: bool = ...,
    ): ...
    def select_as_coordinates(
        self, key: str, where=..., start: int | None = ..., stop: int | None = ...
    ): ...
    def select_column(
        self,
        key: str,
        column: str,
        start: int | None = ...,
        stop: int | None = ...,
    ): ...
    def select_as_multiple(
        self,
        keys,
        where=...,
        selector=...,
        columns=...,
        start=...,
        stop=...,
        iterator=...,
        chunksize=...,
        auto_close: bool = ...,
    ): ...
    def put(
        self,
        key: str,
        value: NDFrame,
        format=...,
        index=...,
        append=...,
        complib=...,
        complevel: int | None = ...,
        min_itemsize: int | dict[str, int] | None = ...,
        nan_rep=...,
        data_columns: list[str] | None = ...,
        encoding=...,
        errors: str = ...,
    ): ...
    def remove(self, key: str, where=..., start=..., stop=...): ...
    def append(
        self,
        key: str,
        value: NDFrame,
        format=...,
        axes=...,
        index=...,
        append=...,
        complib=...,
        complevel: int | None = ...,
        columns=...,
        min_itemsize: int | dict[str, int] | None = ...,
        nan_rep=...,
        chunksize=...,
        expectedrows=...,
        dropna: bool | None = ...,
        data_columns: list[str] | None = ...,
        encoding=...,
        errors: str = ...,
    ): ...
    def append_to_multiple(
        self, d: dict, value, selector, data_columns=..., axes=..., dropna=..., **kwargs
    ): ...
    def create_table_index(
        self,
        key: str,
        columns=...,
        optlevel: int | None = ...,
        kind: str | None = ...,
    ): ...
    def groups(self): ...
    def walk(self, where: str = ...) -> None: ...
    def get_node(self, key: str) -> Node | None: ...
    def get_storer(self, key: str) -> GenericFixed | Table: ...
    def copy(
        self,
        file,
        mode=...,
        propindexes: bool = ...,
        keys=...,
        complib=...,
        complevel: int | None = ...,
        fletcher32: bool = ...,
        overwrite=...,
    ): ...
    def info(self) -> str: ...

class IndexCol:
    is_an_indexable: bool
    is_data_indexable: bool
    name: str
    cname: str
    values = ...  # Incomplete
    kind = ...  # Incomplete
    typ = ...  # Incomplete
    axis = ...  # Incomplete
    pos = ...  # Incomplete
    freq = ...  # Incomplete
    tz = ...  # Incomplete
    index_name = ...  # Incomplete
    ordered = ...  # Incomplete
    table = ...  # Incomplete
    meta = ...  # Incomplete
    metadata = ...  # Incomplete
    def __init__(
        self,
        name: str,
        values: Incomplete | None = ...,
        kind: Incomplete | None = ...,
        typ: Incomplete | None = ...,
        cname: str | None = ...,
        axis: Incomplete | None = ...,
        pos: Incomplete | None = ...,
        freq: Incomplete | None = ...,
        tz: Incomplete | None = ...,
        index_name: Incomplete | None = ...,
        ordered: Incomplete | None = ...,
        table: Incomplete | None = ...,
        meta: Incomplete | None = ...,
        metadata: Incomplete | None = ...,
    ) -> None: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def kind_attr(self) -> str: ...
    def set_pos(self, pos: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other) -> bool: ...
    @property
    def is_indexed(self) -> bool: ...
    def convert(
        self, values: np.ndarray, nan_rep, encoding: str, errors: str
    ) -> tuple[np.ndarray, np.ndarray] | tuple[DatetimeIndex, DatetimeIndex]: ...
    def take_data(self): ...
    @property
    def attrs(self): ...
    @property
    def description(self): ...
    @property
    def col(self): ...
    @property
    def cvalues(self): ...
    def __iter__(self): ...
    def maybe_set_size(self, min_itemsize: Incomplete | None = ...) -> None: ...
    def validate_names(self) -> None: ...
    def validate_and_set(self, handler: AppendableTable, append: bool) -> None: ...
    def validate_col(self, itemsize: Incomplete | None = ...): ...
    def validate_attr(self, append: bool) -> None: ...
    def update_info(self, info) -> None: ...
    def set_info(self, info) -> None: ...
    def set_attr(self) -> None: ...
    def validate_metadata(self, handler: AppendableTable) -> None: ...
    def write_metadata(self, handler: AppendableTable) -> None: ...

class DataCol(IndexCol):
    is_an_indexable: bool
    is_data_indexable: bool
    dtype = ...  # Incomplete
    data = ...  # Incomplete
    def __init__(
        self,
        name: str,
        values: Incomplete | None = ...,
        kind: Incomplete | None = ...,
        typ: Incomplete | None = ...,
        cname: Incomplete | None = ...,
        pos: Incomplete | None = ...,
        tz: Incomplete | None = ...,
        ordered: Incomplete | None = ...,
        table: Incomplete | None = ...,
        meta: Incomplete | None = ...,
        metadata: Incomplete | None = ...,
        dtype: DtypeArg | None = ...,
        data: Incomplete | None = ...,
    ) -> None: ...
    @property
    def dtype_attr(self) -> str: ...
    @property
    def meta_attr(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    kind = ...  # Incomplete
    def set_data(self, data: ArrayLike) -> None: ...
    def take_data(self): ...
    @classmethod
    def get_atom_string(cls, shape, itemsize): ...
    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]: ...
    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col: ...
    @classmethod
    def get_atom_datetime64(cls, shape): ...
    @classmethod
    def get_atom_timedelta64(cls, shape): ...
    @property
    def shape(self): ...
    @property
    def cvalues(self): ...
    def validate_attr(self, append) -> None: ...
    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str): ...
    def set_attr(self) -> None: ...

class Fixed:
    pandas_kind: str
    format_type: str
    obj_type: type[DataFrame | Series]
    ndim: int
    encoding: str
    parent: HDFStore
    group: Node
    errors: str
    is_table: bool
    def __init__(
        self, parent: HDFStore, group: Node, encoding: str = ..., errors: str = ...
    ) -> None: ...
    @property
    def is_old_version(self) -> bool: ...
    @property
    def version(self) -> tuple[int, int, int]: ...
    @property
    def pandas_type(self): ...
    def set_object_info(self) -> None: ...
    def copy(self) -> Fixed: ...
    @property
    def shape(self): ...
    @property
    def pathname(self): ...
    @property
    def attrs(self): ...
    def set_attrs(self) -> None: ...
    def get_attrs(self) -> None: ...
    @property
    def storable(self): ...
    @property
    def is_exists(self) -> bool: ...
    @property
    def nrows(self): ...
    def validate(self, other) -> Literal[True] | None: ...
    def validate_version(self, where: Incomplete | None = ...) -> None: ...
    def infer_axes(self) -> bool: ...
    def read(
        self,
        where: Incomplete | None = ...,
        columns: Incomplete | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
    ): ...
    def write(self, obj, **kwargs) -> None: ...
    def delete(
        self,
        where: Incomplete | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
    ) -> None: ...

class GenericFixed(Fixed):
    attributes: list[str]
    def validate_read(self, columns, where) -> None: ...
    @property
    def is_exists(self) -> bool: ...
    def set_attrs(self) -> None: ...
    encoding: str
    errors: str
    def get_attrs(self) -> None: ...
    def write(self, obj, **kwargs) -> None: ...
    def read_array(self, key: str, start: int | None = ..., stop: int | None = ...): ...
    def read_index(
        self, key: str, start: int | None = ..., stop: int | None = ...
    ) -> Index: ...
    def write_index(self, key: str, index: Index) -> None: ...
    def write_multi_index(self, key: str, index: MultiIndex) -> None: ...
    def read_multi_index(
        self, key: str, start: int | None = ..., stop: int | None = ...
    ) -> MultiIndex: ...
    def read_index_node(
        self, node: Node, start: int | None = ..., stop: int | None = ...
    ) -> Index: ...
    def write_array_empty(self, key: str, value: ArrayLike) -> None: ...
    def write_array(
        self, key: str, obj: AnyArrayLike, items: Index | None = ...
    ) -> None: ...

class Table(Fixed):
    pandas_kind: str
    format_type: str
    table_type: str
    levels: int | list[Hashable]
    is_table: bool
    index_axes: list[IndexCol]
    non_index_axes: list[tuple[int, Any]]
    values_axes: list[DataCol]
    data_columns: list
    metadata: list
    info: dict
    nan_rep = ...  # Incomplete
    def __init__(
        self,
        parent: HDFStore,
        group: Node,
        encoding: Incomplete | None = ...,
        errors: str = ...,
        index_axes: Incomplete | None = ...,
        non_index_axes: Incomplete | None = ...,
        values_axes: Incomplete | None = ...,
        data_columns: Incomplete | None = ...,
        info: Incomplete | None = ...,
        nan_rep: Incomplete | None = ...,
    ) -> None: ...
    @property
    def table_type_short(self) -> str: ...
    def __getitem__(self, c: str): ...
    def validate(self, other) -> None: ...
    @property
    def is_multi_index(self) -> bool: ...
    def validate_multiindex(
        self, obj: DataFrame | Series
    ) -> tuple[DataFrame, list[Hashable]]: ...
    @property
    def nrows_expected(self) -> int: ...
    @property
    def is_exists(self) -> bool: ...
    @property
    def storable(self): ...
    @property
    def table(self): ...
    @property
    def dtype(self): ...
    @property
    def description(self): ...
    @property
    def axes(self): ...
    @property
    def ncols(self) -> int: ...
    @property
    def is_transposed(self) -> bool: ...
    @property
    def data_orientation(self) -> tuple[int, ...]: ...
    def queryables(self) -> dict[str, Any]: ...
    def index_cols(self): ...
    def values_cols(self) -> list[str]: ...
    def write_metadata(self, key: str, values: np.ndarray) -> None: ...
    def read_metadata(self, key: str): ...
    def set_attrs(self) -> None: ...
    encoding: str
    errors: str
    def get_attrs(self) -> None: ...
    def validate_version(self, where: Incomplete | None = ...) -> None: ...
    def validate_min_itemsize(self, min_itemsize) -> None: ...
    def indexables(self): ...
    def create_index(
        self,
        columns: Incomplete | None = ...,
        optlevel: Incomplete | None = ...,
        kind: str | None = ...,
    ) -> None: ...
    @classmethod
    def get_object(cls, obj, transposed: bool): ...
    def validate_data_columns(self, data_columns, min_itemsize, non_index_axes): ...
    def process_axes(
        self, obj, selection: Selection, columns: Incomplete | None = ...
    ) -> DataFrame: ...
    def create_description(
        self,
        complib,
        complevel: int | None,
        fletcher32: bool,
        expectedrows: int | None,
    ) -> dict[str, Any]: ...
    def read_coordinates(
        self,
        where: Incomplete | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
    ): ...
    def read_column(
        self,
        column: str,
        where: Incomplete | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
    ): ...

class AppendableTable(Table):
    table_type: str
    # Class design makes this untypable
    def write(  # type: ignore[override]
        self,
        obj,
        axes: Incomplete | None = ...,
        append: bool = ...,
        complib: Incomplete | None = ...,
        complevel: Incomplete | None = ...,
        fletcher32: Incomplete | None = ...,
        min_itemsize: Incomplete | None = ...,
        chunksize: Incomplete | None = ...,
        expectedrows: Incomplete | None = ...,
        dropna: bool = ...,
        nan_rep: Incomplete | None = ...,
        data_columns: Incomplete | None = ...,
        track_times: bool = ...,
    ) -> None: ...
    def write_data(self, chunksize: int | None, dropna: bool = ...) -> None: ...
    def write_data_chunk(
        self,
        rows: np.ndarray,
        indexes: list[np.ndarray],
        mask: npt.NDArray[np.bool_] | None,
        values: list[np.ndarray],
    ) -> None: ...
    def delete(
        self,
        where: Incomplete | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
    ): ...

class Selection:
    table = ...  # Incomplete
    where = ...  # Incomplete
    start = ...  # Incomplete
    stop = ...  # Incomplete
    condition = ...  # Incomplete
    filter = ...  # Incomplete
    terms = ...  # Incomplete
    coordinates = ...  # Incomplete
    def __init__(
        self,
        table: Table,
        where: Incomplete | None = ...,
        start: int | None = ...,
        stop: int | None = ...,
    ) -> None: ...
    def generate(self, where): ...
    def select(self): ...
    def select_coords(self): ...
