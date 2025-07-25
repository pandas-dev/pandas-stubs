from typing import Any

from pandas import DataFrame

from pandas._typing import (
    DtypeBackend,
    FilePath,
    ParquetEngine,
    ReadBuffer,
    StorageOptions,
)

def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: ParquetEngine = ...,
    columns: list[str] | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend = ...,
    filesystem: Any = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    **kwargs: Any,
) -> DataFrame: ...
