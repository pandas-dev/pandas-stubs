from typing import (
    Any,
    Iterator,
    overload,
)

from pandas.core.base import PandasObject
from pandas.core.frame import DataFrame

from pandas._typing import (
    DtypeArg,
    npt,
)

class DatabaseError(IOError): ...

@overload
def read_sql_table(
    table_name: str,
    con: Any,
    schema: str | list[str] = ...,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] | None = ...,
    *,
    chunksize: int,
) -> Iterator[DataFrame]: ...
@overload
def read_sql_table(
    table_name: str,
    con: Any,
    schema: str | list[str] = ...,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] | None = ...,
    chunksize: None = ...,
) -> DataFrame: ...
@overload
def read_sql_query(
    sql: str,
    con: Any,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[str] | dict[str, str] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    *,
    chunksize: int,
    dtype: DtypeArg | None = ...,
) -> Iterator[DataFrame]: ...
@overload
def read_sql_query(
    sql: str,
    con: Any,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[str] | dict[str, str] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
) -> DataFrame: ...
@overload
def read_sql(
    sql: str,
    con: Any,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[str] | dict[str, str] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] = ...,
    *,
    chunksize: int,
) -> Iterator[DataFrame]: ...
@overload
def read_sql(
    sql: str,
    con: Any,
    index_col: str | list[str] | None = ...,
    coerce_float: bool = ...,
    params: list[str] | dict[str, str] | None = ...,
    parse_dates: list[str] | dict[str, str] | dict[str, dict[str, Any]] | None = ...,
    columns: list[str] = ...,
    chunksize: None = ...,
) -> DataFrame: ...

class PandasSQL(PandasObject):
    def read_sql(self, *args, **kwargs): ...
    def to_sql(
        self,
        frame: DataFrame,
        name: str,
        if_exists: str = ...,
        index: bool = ...,
        index_label=...,
        schema=...,
        chunksize=...,
        dtype: DtypeArg | None = ...,
        method=...,
    ) -> int | None: ...

class SQLTable(PandasObject):
    name: str
    pd_sql: PandasSQL  # pandas SQL interface
    prefix: str
    frame: DataFrame | None
    index: list[str]
    schema: str
    if_exists: str
    keys: list[str]
    dtype: DtypeArg | None
    table: Any  # sqlalchemy.Table
    def __init__(
        self,
        name: str,
        pandas_sql_engine: PandasSQL,
        frame: DataFrame | None = ...,
        index: bool | str | list[str] | None = ...,
        if_exists: str = ...,
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
    ) -> DataFrame | Iterator[DataFrame]: ...
