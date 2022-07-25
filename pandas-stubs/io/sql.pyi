from __future__ import annotations

from typing import (
    Any,
    Callable,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    overload,
)

from pandas.core.frame import DataFrame

from pandas._typing import DtypeArg

class SQLAlchemyRequired(ImportError): ...
class DatabaseError(IOError): ...

@overload
def read_sql_table(
    table_name: str,
    con: str | Any,
    schema: str | None = ...,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates: Sequence[str] | Mapping[str, str] | None = ...,
    columns: Sequence[str] | None = ...,
    chunksize: None = ...,
) -> DataFrame: ...
@overload
def read_sql_table(
    table_name: str,
    con: str | Any,
    schema: str | None = ...,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates: Sequence[str] | Mapping[str, str] | None = ...,
    columns: Sequence[str] | None = ...,
    *,
    chunksize: int,
) -> Iterator[DataFrame]: ...
@overload
def read_sql_query(
    sql: str | Any,
    con: str | Any,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    params=...,
    parse_dates: Sequence[str] | Mapping[str, str] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
) -> DataFrame: ...
@overload
def read_sql_query(
    sql: str | Any,
    con: str | Any,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    params=...,
    parse_dates: Sequence[str] | Mapping[str, str] | None = ...,
    *,
    chunksize: int,
    dtype: DtypeArg | None = ...,
) -> Iterator[DataFrame]: ...
@overload
def read_sql(
    sql: str | Any,
    con: str | Any = ...,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    params: Sequence[str] | tuple[str, ...] | Mapping[str, str] | None = ...,
    parse_dates: Sequence[str]
    | Mapping[str, str]
    | Mapping[str, Mapping[str, Any]]
    | None = ...,
    columns: Sequence[str] = ...,
    chunksize: None = ...,
) -> DataFrame: ...
@overload
def read_sql(
    sql: str | Any,
    con: str | Any = ...,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    params: Sequence[str] | tuple[str, ...] | Mapping[str, str] | None = ...,
    parse_dates: Sequence[str]
    | Mapping[str, str]
    | Mapping[str, Mapping[str, Any]]
    | None = ...,
    columns: Sequence[str] = ...,
    *,
    chunksize: int,
) -> Iterator[DataFrame]: ...
def to_sql(
    frame: DataFrame,
    name: str,
    con: Any,
    schema: str | None = ...,
    if_exists: str = ...,
    index: bool = ...,
    index_label: Hashable | Sequence[Hashable] | None = ...,
    chunksize: int | None = ...,
    dtype: DtypeArg | None = ...,
    method: str | Callable | None = ...,
    engine: str = ...,
    **engine_kwargs,
) -> int | None: ...
