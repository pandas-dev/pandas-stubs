from typing import (
    Any,
    Iterator,
    overload,
)

from _typeshed import Incomplete
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

class SQLTable(PandasObject):
    name: Incomplete
    pd_sql: Incomplete
    prefix: Incomplete
    frame: Incomplete
    index: Incomplete
    schema: Incomplete
    if_exists: Incomplete
    keys: Incomplete
    dtype: Incomplete
    table: Incomplete
    def __init__(
        self,
        name: str,
        pandas_sql_engine,
        frame: Incomplete | None = ...,
        index: bool | str | list[str] | None = ...,
        if_exists: str = ...,
        prefix: str = ...,
        index_label: Incomplete | None = ...,
        schema: Incomplete | None = ...,
        keys: Incomplete | None = ...,
        dtype: DtypeArg | None = ...,
    ) -> None: ...
    def exists(self): ...
    def sql_schema(self) -> str: ...
    def create(self) -> None: ...
    def insert_data(self) -> tuple[list[str], list[npt.NDArray]]: ...
    def insert(
        self, chunksize: int | None = ..., method: str | None = ...
    ) -> int | None: ...
    def read(
        self,
        coerce_float: bool = ...,
        parse_dates: Incomplete | None = ...,
        columns: Incomplete | None = ...,
        chunksize: Incomplete | None = ...,
    ) -> DataFrame | Iterator[DataFrame]: ...
