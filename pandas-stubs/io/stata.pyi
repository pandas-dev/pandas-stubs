from collections import abc
import datetime
from io import BytesIO
from types import TracebackType
from typing import (
    Literal,
    Sequence,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.core.frame import DataFrame

from pandas._typing import (
    CompressionOptions,
    FilePath,
    HashableT,
    ReadBuffer,
    StataDateFormat,
    StorageOptions,
    WriteBuffer,
)

@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: int | None = ...,
    iterator: Literal[True],
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> StataReader: ...
@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool,
    convert_categoricals: bool,
    index_col: str | None,
    convert_missing: bool,
    preserve_dtypes: bool,
    columns: list[HashableT] | None,
    order_categoricals: bool,
    chunksize: int,
    iterator: bool = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> StataReader: ...
@overload
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: None = ...,
    iterator: Literal[False] = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...

class StataParser:
    def __init__(self) -> None: ...

class StataReader(StataParser, abc.Iterator):
    col_sizes: list[int] = ...
    path_or_buf: BytesIO = ...
    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        convert_dates: bool = ...,
        convert_categoricals: bool = ...,
        index_col: str | None = ...,
        convert_missing: bool = ...,
        preserve_dtypes: bool = ...,
        columns: Sequence[str] | None = ...,
        order_categoricals: bool = ...,
        chunksize: int | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def __enter__(self) -> StataReader: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...
    def __next__(self) -> DataFrame: ...
    def get_chunk(self, size: int | None = ...) -> DataFrame: ...
    def read(
        self,
        nrows: int | None = ...,
        convert_dates: bool | None = ...,
        convert_categoricals: bool | None = ...,
        index_col: str | None = ...,
        convert_missing: bool | None = ...,
        preserve_dtypes: bool | None = ...,
        columns: list[str] | None = ...,
        order_categoricals: bool | None = ...,
    ): ...
    @property
    def data_label(self) -> str: ...
    def variable_labels(self) -> dict[str, str]: ...
    def value_labels(self) -> dict[str, dict[float, str]]: ...

class StataWriter(StataParser):
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[HashableT, StataDateFormat] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[HashableT, str] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        *,
        value_labels: dict[HashableT, dict[float, str]] | None = ...,
    ) -> None: ...
    def write_file(self) -> None: ...
