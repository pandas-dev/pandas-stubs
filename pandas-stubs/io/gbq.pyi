from typing import Any

from pandas.core.frame import DataFrame

def read_gbq(
    query: str,
    project_id: str | None = ...,
    index_col: str | None = ...,
    col_order: list[str] | None = ...,
    reauth: bool = ...,
    auth_local_webserver: bool = ...,
    dialect: str | None = ...,
    location: str | None = ...,
    configuration: dict[str, Any] | None = ...,
    credentials=...,
    use_bqstorage_api: bool | None = ...,
    private_key=...,
    verbose=...,
    progress_bar_type: str | None = ...,
) -> DataFrame: ...
def to_gbq(
    dataframe: DataFrame,
    destination_table: str,
    project_id: str | None = ...,
    chunksize: int | None = ...,
    reauth: bool = ...,
    if_exists: str = ...,
    auth_local_webserver: bool = ...,
    table_schema: list[dict[str, str]] | None = ...,
    location: str | None = ...,
    progress_bar: bool = ...,
    credentials=...,
    verbose=...,
    private_key=...,
) -> None: ...
