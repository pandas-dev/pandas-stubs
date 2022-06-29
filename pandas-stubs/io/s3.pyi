from typing import (
    IO,
    Any,
    Optional,
    Tuple,
)

from pandas._typing import FilePathOrBuffer as FilePathOrBuffer

s3fs = ...

def get_file_and_filesystem(
    filepath_or_buffer: FilePathOrBuffer, mode: Optional[str] = ...
) -> Tuple[IO, Any]: ...
def get_filepath_or_buffer(
    filepath_or_buffer: FilePathOrBuffer,
    encoding: Optional[str] = ...,
    compression: Optional[str] = ...,
    mode: Optional[str] = ...,
) -> Tuple[IO, Optional[str], Optional[str], bool]: ...
