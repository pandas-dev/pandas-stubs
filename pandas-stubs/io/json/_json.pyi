from collections import abc
from collections.abc import Mapping
from types import TracebackType
from typing import (
    Generic,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._libs.lib import NoDefault
from pandas._typing import (
    CompressionOptions,
    DtypeArg,
    DtypeBackend,
    FilePath,
    HashableT,
    JsonFrameOrient,
    JsonSeriesOrient,
    NDFrameT,
    ReadBuffer,
    StorageOptions,
    TimeUnit,
)

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = ...,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: bool,
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = ...,
) -> JsonReader[Series]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = ...,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"] = ...,
) -> JsonReader[Series]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = ...,
    typ: Literal["frame"] = ...,
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"] = ...,
) -> JsonReader[DataFrame]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = ...,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = ...,
) -> Series: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = ...,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: Literal[True] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"] = ...,
) -> Series: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = ...,
    typ: Literal["frame"] = ...,
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = ...,
) -> DataFrame: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = ...,
    typ: Literal["frame"] = ...,
    dtype: bool | Mapping[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: TimeUnit | None = ...,
    encoding: str | None = ...,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = ...,
    lines: Literal[True] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"] = ...,
) -> DataFrame: ...

class JsonReader(abc.Iterator, Generic[NDFrameT]):
    def read(self) -> NDFrameT: ...
    def close(self) -> None: ...
    def __iter__(self) -> JsonReader[NDFrameT]: ...
    def __next__(self) -> NDFrameT: ...
    def __enter__(self) -> JsonReader[NDFrameT]: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
