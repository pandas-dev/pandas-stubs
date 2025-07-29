from typing import Any

from pandas._typing import (
    CompressionOptions,
    FilePath,
    ReadPickleBuffer,
    StorageOptions,
    WriteBuffer,
)

def to_pickle(
    obj: object,
    filepath_or_buffer: FilePath | WriteBuffer[bytes],
    compression: CompressionOptions = "infer",
    protocol: int = ...,
    storage_options: StorageOptions = None,
) -> None: ...
def read_pickle(
    filepath_or_buffer: FilePath | ReadPickleBuffer,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions = None,
) -> Any: ...
