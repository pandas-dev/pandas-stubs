from __future__ import annotations

from typing import (
    Any,
    Callable,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame
from pandas.core.series import Series

from pandas._libs.json import (
    dumps as dumps,
    loads as loads,
)
from pandas._typing import (
    CompressionOptions,
    DtypeArg,
    FilePath,
    JSONSerializable,
    ReadBuffer,
    StorageOptions,
    WriteBuffer,
)

def to_json(
    path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes],
    obj: DataFrame | Series,
    orient: str | None = ...,
    date_format: Literal["epoch", "iso"] | None = ...,
    double_precision: int = ...,
    force_ascii: bool = ...,
    date_unit: str = ...,
    default_handler: Callable[[Any], JSONSerializable] | None = ...,
    lines: bool = ...,
    compression: CompressionOptions = ...,
    index: bool = ...,
    indent: int = ...,
    storage_options: StorageOptions = ...,
) -> str | None: ...
@overload
def read_json(
    path: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    orient: str | None,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool = ...,
    keep_default_dates: bool = ...,
    # Removed since deprecated
    # numpy: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ] = ...,
    lines: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> Series: ...
@overload
def read_json(
    path: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    *,
    orient: str | None = ...,
    typ: Literal["series"],
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool = ...,
    keep_default_dates: bool = ...,
    # Removed since deprecated
    # numpy: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ] = ...,
    lines: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> Series: ...
@overload
def read_json(
    path: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
    orient: str | None = ...,
    typ: Literal["frame"] = ...,
    dtype: DtypeArg | None = ...,
    convert_axes: bool | None = ...,
    convert_dates: bool = ...,
    keep_default_dates: bool = ...,
    # Removed since deprecated
    # numpy: bool = ...,
    precise_float: bool = ...,
    date_unit: str | None = ...,
    encoding: Literal[
        "strict", "ignore", "replace", "backslashreplace", "surrogateescape"
    ] = ...,
    lines: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    nrows: int | None = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...
