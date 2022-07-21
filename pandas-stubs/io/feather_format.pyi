from __future__ import annotations

from pandas.core.frame import DataFrame

from pandas._typing import (
    FilePath,
    HashableT,
    ReadBuffer,
    StorageOptions,
    WriteBuffer,
)

def to_feather(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes],
    storage_options: StorageOptions = ...,
    **kwargs,
) -> None: ...
def read_feather(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = ...,
    use_threads: bool = ...,
    storage_options: StorageOptions = ...,
): ...
