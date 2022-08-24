from typing import (
    Any,
    Literal,
)

from pandas import DataFrame

from pandas._typing import (
    FilePath,
    ReadBuffer,
    StorageOptions,
    ParquetEngine,
)

def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: ParquetEngine = ...,
    columns: list[str] | None = ...,
    storage_options: StorageOptions = ...,
    use_nullable_dtypes: bool = ...,
    **kwargs: Any,
) -> DataFrame: ...
