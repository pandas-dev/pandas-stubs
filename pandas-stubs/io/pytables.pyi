from typing import (
    Any,
    Literal,
    overload,
)

import numpy as np
from pandas import (
    DataFrame,
    Index,
    Series,
)
from pandas.core.computation.pytables import PyTablesExpr
from pandas.core.generic import NDFrame

from pandas._typing import (
    FilePath,
    HashableT,
)

Term = PyTablesExpr

class PossibleDataLossError(Exception): ...
class ClosedFileError(Exception): ...
class IncompatibilityWarning(Warning): ...
class AttributeConflictWarning(Warning): ...
class DuplicateWarning(Warning): ...

@overload
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key: Any | None = ...,
    mode: Literal["r", "r+", "a"] = ...,
    errors: Literal[
        "strict",
        "ignore",
        "replace",
        "surrogateescape",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
    ]
    | None = ...,
    where: list[Term] | str | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[HashableT] | None = ...,
    *,
    iterator: Literal[True],
    chunksize: int | None = ...,
    **kwargs: Any,
) -> TableIterator: ...
@overload
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key: Any | None = ...,
    mode: Literal["r", "r+", "a"] = ...,
    errors: Literal[
        "strict",
        "ignore",
        "replace",
        "surrogateescape",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
    ]
    | None = ...,
    where: list[Term] | str | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[HashableT] | None = ...,
    iterator: bool = ...,
    *,
    chunksize: int,
    **kwargs: Any,
) -> TableIterator: ...
@overload
def read_hdf(
    path_or_buf: FilePath | HDFStore,
    key: Any | None = ...,
    mode: Literal["r", "r+", "a"] = ...,
    errors: Literal[
        "strict",
        "ignore",
        "replace",
        "surrogateescape",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
    ]
    | None = ...,
    where: list[Term] | str | None = ...,
    start: int | None = ...,
    stop: int | None = ...,
    columns: list[HashableT] | None = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    **kwargs: Any,
) -> DataFrame | Series | Index: ...

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
    def open(self, mode: str = ..., **kwargs): ...
    def close(self) -> None: ...
    @property
    def is_open(self) -> bool: ...
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
    def groups(self): ...
    def walk(self, where: str = ...) -> None: ...
    def info(self) -> str: ...

class TableIterator:
    def __iter__(self): ...
    def close(self) -> None: ...
