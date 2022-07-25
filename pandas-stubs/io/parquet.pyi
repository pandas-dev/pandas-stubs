from __future__ import annotations

from types import ModuleType
from typing import (
    Any,
    Literal,
)

from pandas.core.frame import DataFrame

from pandas._typing import (
    FilePath,
    FilePathOrBuffer,
    ReadBuffer,
    StorageOptions,
    WriteBuffer,
)

_EngineT = Literal["auto", "pyarrow", "fastparquet"]
_CompressionT = Literal["snappy", "gzip", "brotli", "lz4", "zstd"]

def get_engine(engine: _EngineT) -> BaseImpl: ...

class BaseImpl:
    @staticmethod
    def validate_dataframe(df: DataFrame) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: _CompressionT | None,
        **kwargs: Any,
    ): ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        **kwargs: Any,
    ) -> None: ...

class PyArrowImpl(BaseImpl):
    api: ModuleType = ...
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: _CompressionT | None = ...,
        index: bool | None = ...,
        storage_options: StorageOptions = ...,
        partition_cols: list[str] | None = ...,
        **kwargs,
    ): ...
    def read(
        self,
        path: FilePath | ReadBuffer[bytes],
        columns: list[str] | None = ...,
        use_nullable_dtypes: bool = ...,
        storage_options: StorageOptions = ...,
        **kwargs,
    ): ...

class FastParquetImpl(BaseImpl):
    api: ModuleType = ...
    def __init__(self) -> None: ...
    def write(
        self,
        df: DataFrame,
        path: FilePath | WriteBuffer[bytes],
        compression: _CompressionT | None = ...,
        index: bool | None = ...,
        partition_cols: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs: Any,
    ): ...
    def read(
        self,
        path: FilePath | WriteBuffer[bytes],
        columns: list[str] | None = ...,
        storage_options: StorageOptions = ...,
        **kwargs: Any,
    ): ...

def to_parquet(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None,
    engine: _EngineT = ...,
    compression: _CompressionT | None = ...,
    index: bool | None = ...,
    storage_options: StorageOptions = ...,
    partition_cols: list[str] | None = ...,
    **kwargs: Any,
) -> bytes | None: ...
def read_parquet(
    path: FilePathOrBuffer,
    engine: _EngineT = ...,
    columns: list[str] | None = ...,
    storage_options: StorageOptions = ...,
    use_nullable_dtypes: bool = ...,
    **kwargs: Any,
) -> DataFrame: ...
