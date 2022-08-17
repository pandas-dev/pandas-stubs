from collections import abc
import datetime
from io import BytesIO
from types import TracebackType
from typing import (
    Hashable,
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
    StorageOptions,
    WriteBuffer,
)

@overload
def read_stata(
    path: FilePath | ReadBuffer[bytes],
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: int | None = ...,
    *,
    iterator: Literal[True],
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> StataReader: ...
@overload
def read_stata(
    path: FilePath | ReadBuffer[bytes],
    convert_dates: bool,
    convert_categoricals: bool,
    index_col: str | None,
    convert_missing: bool,
    preserve_dtypes: bool,
    columns: list[HashableT] | None,
    order_categoricals: bool,
    chunksize: int | None,
    iterator: Literal[True],
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> StataReader: ...
@overload
def read_stata(
    path: FilePath | ReadBuffer[bytes],
    convert_dates: bool = ...,
    convert_categoricals: bool = ...,
    index_col: str | None = ...,
    convert_missing: bool = ...,
    preserve_dtypes: bool = ...,
    columns: list[HashableT] | None = ...,
    order_categoricals: bool = ...,
    chunksize: int | None = ...,
    iterator: Literal[False] = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...

class PossiblePrecisionLoss(Warning): ...
class ValueLabelTypeMismatch(Warning): ...
class InvalidColumnName(Warning): ...

class StataParser:
    DTYPE_MAP: dict[int, np.dtype] = ...
    DTYPE_MAP_XML: dict[int, np.dtype] = ...
    TYPE_MAP: list[tuple[int | str, ...]] = ...
    TYPE_MAP_XML: dict[int, str] = ...
    VALID_RANGE: dict[
        str,
        tuple[int, int] | tuple[np.float32, np.float32] | tuple[np.float64, np.float64],
    ] = ...
    OLD_TYPE_MAPPING: dict[int, int] = ...
    MISSING_VALUES: dict[str, int] = ...
    NUMPY_TYPE_MAP: dict[str, str] = ...
    RESERVED_WORDS: tuple[str, ...] = ...
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
    type_converters: dict[str, type[np.dtype]] = ...
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = ...,
    ) -> None: ...
    def write_file(self) -> None: ...

class StataWriter117(StataWriter):
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        convert_strl: Sequence[Hashable] | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = ...,
    ) -> None: ...

class StataWriterUTF8(StataWriter117):
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = ...,
        write_index: bool = ...,
        byteorder: str | None = ...,
        time_stamp: datetime.datetime | None = ...,
        data_label: str | None = ...,
        variable_labels: dict[Hashable, str] | None = ...,
        convert_strl: Sequence[Hashable] | None = ...,
        version: int | None = ...,
        compression: CompressionOptions = ...,
        storage_options: StorageOptions = ...,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = ...,
    ) -> None: ...
