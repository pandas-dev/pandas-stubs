from __future__ import annotations

from abc import (
    ABC,
    ABCMeta,
    abstractmethod,
)
from io import (
    BytesIO,
    StringIO,
    TextIOBase,
)
from pathlib import Path
import tarfile
from typing import (
    IO,
    AnyStr,
    Generic,
    Literal,
    TypeVar,
    overload,
)
import zipfile

from pandas._typing import (
    BaseBuffer,
    CompressionDict,
    CompressionOptions,
    FilePath,
    ReadBuffer,
    StorageOptions,
    WriteBuffer,
)

_BaseBufferT = TypeVar("_BaseBufferT", bound=BaseBuffer)

class IOArgs:
    filepath_or_buffer: str | BaseBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool
    def __init__(
        self, filepath_or_buffer, encoding, mode, compression, should_close
    ) -> None: ...

class IOHandles(Generic[AnyStr]):
    handle: IO[AnyStr]
    compression: CompressionDict
    created_handles: list[IO[bytes] | IO[str]]
    is_wrapped: bool
    def close(self) -> None: ...
    def __enter__(self) -> IOHandles[AnyStr]: ...
    def __exit__(self, *args: object) -> None: ...
    def __init__(self, handle, compression, created_handles, is_wrapped) -> None: ...

def is_url(url: object) -> bool: ...
def validate_header_arg(header: object) -> None: ...
@overload
def stringify_path(
    filepath_or_buffer: FilePath, convert_file_like: bool = ...
) -> str: ...
@overload
def stringify_path(
    filepath_or_buffer: _BaseBufferT, convert_file_like: bool = ...
) -> _BaseBufferT: ...
def urlopen(*args, **kwargs): ...
def is_fsspec_url(url: FilePath | BaseBuffer) -> bool: ...
def file_path_to_url(path: str) -> str: ...
def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]: ...
def infer_compression(
    filepath_or_buffer: FilePath | BaseBuffer, compression: str | None
) -> str | None: ...
def check_parent_directory(path: Path | str) -> None: ...
@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[False],
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[bytes]: ...
@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: Literal[True] = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str]: ...
@overload
def get_handle(
    path_or_buf: FilePath | BaseBuffer,
    mode: str,
    *,
    encoding: str | None = ...,
    compression: CompressionOptions = ...,
    memory_map: bool = ...,
    is_text: bool = ...,
    errors: str | None = ...,
    storage_options: StorageOptions = ...,
) -> IOHandles[str] | IOHandles[bytes]: ...

class _BufferedWriter(BytesIO, ABC, metaclass=ABCMeta):
    @abstractmethod
    def write_to_buffer(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> _BufferedWriter: ...

class _BytesTarFile(_BufferedWriter):
    archive_name: str | None
    name: str
    buffer: tarfile.TarFile
    def __init__(
        self,
        name: str | None = ...,
        mode: Literal["r", "a", "w", "x"] = ...,
        fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None = ...,
        archive_name: str | None = ...,
        **kwargs,
    ) -> None: ...
    def extend_mode(self, mode: str) -> str: ...
    def infer_filename(self) -> str | None: ...
    def write_to_buffer(self) -> None: ...

class _BytesZipFile(_BufferedWriter):
    archive_name: str | None
    buffer: zipfile.ZipFile
    def __init__(
        self,
        file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
        mode: str,
        archive_name: str | None = ...,
        **kwargs,
    ) -> None: ...
    def infer_filename(self) -> str | None: ...
    def write_to_buffer(self) -> None: ...

class _IOWrapper:
    buffer: BaseBuffer
    def __init__(self, buffer: BaseBuffer) -> None: ...
    def __getattr__(self, name: str): ...
    def readable(self) -> bool: ...
    def seekable(self) -> bool: ...
    def writable(self) -> bool: ...

class _BytesIOWrapper:
    buffer: StringIO | TextIOBase
    encoding: str
    overflow: bytes
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = ...) -> None: ...
    def __getattr__(self, attr: str): ...
    def read(self, n: int | None = ...) -> bytes: ...

def file_exists(filepath_or_buffer: FilePath | BaseBuffer) -> bool: ...
