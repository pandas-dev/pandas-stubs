from collections.abc import (
    Iterator,
    Mapping,
)
from types import TracebackType
from typing import (
    Any,
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
    orient: JsonSeriesOrient | None = None,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = "ujson",
) -> JsonReader[Series]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = None,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"],
) -> JsonReader[Series]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = None,
    typ: Literal["frame"] = "frame",
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = "ujson",
) -> JsonReader[DataFrame]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = None,
    typ: Literal["frame"] = "frame",
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"],
) -> JsonReader[DataFrame]: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = None,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: bool = False,
    chunksize: None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = "ujson",
) -> Series: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = None,
    typ: Literal["series"],
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = False,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: Literal[True],
    chunksize: None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"],
) -> Series: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = None,
    typ: Literal["frame"] = "frame",
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = False,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: bool = False,
    chunksize: None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["ujson"] = "ujson",
) -> DataFrame: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = None,
    typ: Literal["frame"] = "frame",
    dtype: bool | Mapping[HashableT, DtypeArg] | None = None,
    convert_axes: bool | None = False,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: TimeUnit | None = None,
    encoding: str | None = None,
    encoding_errors: (
        Literal["strict", "ignore", "replace", "backslashreplace", "surrogateescape"]
        | None
    ) = "strict",
    lines: Literal[True],
    chunksize: None = None,
    compression: CompressionOptions = "infer",
    nrows: int | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend | NoDefault = ...,
    engine: Literal["pyarrow"],
) -> DataFrame: ...

class JsonReader(Iterator[Any], Generic[NDFrameT]):
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
