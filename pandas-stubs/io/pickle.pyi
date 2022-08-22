from pandas._typing import (
    CompressionOptions,
    FilePath,
    StorageOptions,
    WriteBuffer,
)

def to_pickle(
    obj: object,
    filepath_or_buffer: FilePath | WriteBuffer[bytes],
    compression: CompressionOptions = ...,
    protocol: int = ...,
    storage_options: StorageOptions = ...,
) -> None: ...
