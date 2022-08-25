from collections import abc
from typing import (
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._typing import (
    CompressionOptions,
    DtypeArg,
    FilePath,
    HashableT,
    JsonFrameOrient,
    JsonSeriesOrient,
    ReadBuffer,
    StorageOptions,
)

@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = ...,
    typ: Literal["series"],
    dtype: bool | dict[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Literal["s", "ms", "us", "ns"] | None = ...,
    encoding: str | None = ...,
    encoding_errors: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ]
    | None = ...,
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> JsonReader: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonFrameOrient | None = ...,
    typ: Literal["frame"] = ...,
    dtype: bool | dict[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Literal["s", "ms", "us", "ns"] | None = ...,
    encoding: str | None = ...,
    encoding_errors: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ]
    | None = ...,
    lines: Literal[True],
    chunksize: int,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> JsonReader: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: JsonSeriesOrient | None = ...,
    typ: Literal["series"],
    dtype: bool | dict[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Literal["s", "ms", "us", "ns"] | None = ...,
    encoding: str | None = ...,
    encoding_errors: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ]
    | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> Series: ...
@overload
def read_json(
    path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    orient: JsonFrameOrient | None = ...,
    typ: Literal["frame"] = ...,
    dtype: bool | dict[HashableT, DtypeArg] | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool | list[str] = ...,
    keep_default_dates: bool = ...,
    precise_float: bool = ...,
    date_unit: Literal["s", "ms", "us", "ns"] | None = ...,
    encoding: str | None = ...,
    encoding_errors: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ]
    | None = ...,
    lines: bool = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...

class JsonReader(abc.Iterator):
    def read(self) -> DataFrame | Series: ...
    def close(self) -> None: ...
    def __iter__(self) -> JsonReader: ...
    def __next__(self) -> DataFrame | Series: ...
    def __enter__(self) -> JsonReader: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
