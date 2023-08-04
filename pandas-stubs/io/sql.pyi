from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Mapping,
)
import sqlite3
from typing import (
    Any,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
import sqlalchemy.engine
from sqlalchemy.orm import FromStatement
import sqlalchemy.sql.expression
from typing_extensions import TypeAlias

from pandas._libs.lib import NoDefault
from pandas._typing import (
    DtypeArg,
    DtypeBackend,
    Scalar,
    npt,
)

_SQLConnection: TypeAlias = str | sqlalchemy.engine.Connectable | sqlite3.Connection

_SQLStatement: TypeAlias = (
    str
    | sqlalchemy.sql.expression.Selectable
    | sqlalchemy.sql.expression.TextClause
    | sqlalchemy.sql.Select
    | FromStatement
)

@overload
def read_sql_table(
    table_name: str,
    con: _SQLConnection,
    schema: str | None = ...,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] | None = ...,
    *,
    chunksize: int,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> Generator[DataFrame, None, None]: ...
@overload
def read_sql_table(
    table_name: str,
    con: _SQLConnection,
    schema: str | None = ...,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] | None = ...,
    chunksize: None = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> DataFrame: ...
@overload
def read_sql_query(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[Scalar] | tuple[Scalar, ...] | Mapping[str, Scalar] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    *,
    chunksize: int,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> Generator[DataFrame, None, None]: ...
@overload
def read_sql_query(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[Scalar] | tuple[Scalar, ...] | Mapping[str, Scalar] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> DataFrame: ...
@overload
def read_sql(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[Scalar] | tuple[Scalar, ...] | Mapping[str, Scalar] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] = ...,
    *,
    chunksize: int,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> Generator[DataFrame, None, None]: ...
@overload
def read_sql(
    sql: _SQLStatement,
    con: _SQLConnection,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[Scalar] | tuple[Scalar, ...] | Mapping[str, Scalar] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
) -> DataFrame: ...

class PandasSQL:
    def read_sql(self, *args, **kwargs): ...
    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: Literal["fail", "replace", "append"] = ...,
        index: bool = ...,
        index_label=...,
        schema: str | None = ...,
        chunksize=...,
        dtype: DtypeArg | None = ...,
        method: Literal["multi"]
        | Callable[[SQLTable, Any, list[str], Iterable], int | None]
        | None = ...,
    ) -> int | None: ...

class SQLTable:
    name: str
    pd_sql: PandasSQL  # pandas SQL interface
    prefix: str
    frame: DataFrame | None
    index: list[str]
    schema: str
    if_exists: Literal["fail", "replace", "append"]
    keys: list[str]
    dtype: DtypeArg | None
    table: Any  # sqlalchemy.Table
    def __init__(
        self,
        name: str,
        pandas_sql_engine: PandasSQL,
        frame: DataFrame | None = ...,
        index: bool | str | list[str] | None = ...,
        if_exists: Literal["fail", "replace", "append"] = ...,
        prefix: str = ...,
        index_label: str | list[str] | None = ...,
        schema: str | None = ...,
        keys: str | list[str] | None = ...,
        dtype: DtypeArg | None = ...,
    ) -> None: ...
    def exists(self) -> bool: ...
    def sql_schema(self) -> str: ...
    def create(self) -> None: ...
    def insert_data(self) -> tuple[list[str], list[npt.NDArray]]: ...
    def insert(
        self, chunksize: int | None = ..., method: str | None = ...
    ) -> int | None: ...
    def read(
        self,
        coerce_float: bool = ...,
        parse_dates: bool | list[str] | None = ...,
        columns: list[str] | None = ...,
        chunksize: int | None = ...,
    ) -> DataFrame | Generator[DataFrame, None, None]: ...
