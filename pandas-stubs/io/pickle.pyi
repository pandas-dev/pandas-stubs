from pandas._typing import (
    CompressionOptions,
    FilePathOrBuffer,
)

def to_pickle(
    obj,
    filepath_or_buffer: FilePathOrBuffer,
    compression: str | None = ...,
    protocol: int = ...,
): ...
def read_pickle(
    filepath_or_buffer_or_reader: FilePathOrBuffer,
    compression: CompressionOptions = ...,
): ...
