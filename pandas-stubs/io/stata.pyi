from __future__ import annotations

from collections import abc
import datetime
from io import BytesIO
from types import TracebackType
from typing import (
    Hashable,
    Literal,
    Sequence,
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
    iterator: bool = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame | StataReader: ...

stata_epoch: datetime.datetime = ...
excessive_string_length_error: str

class PossiblePrecisionLoss(Warning): ...

precision_loss_doc: str

class ValueLabelTypeMismatch(Warning): ...

value_label_mismatch_doc: str

class InvalidColumnName(Warning): ...

invalid_name_doc: str

class StataValueLabel:
    labname: Hashable = ...
    value_labels: list[tuple[int | float, str]] = ...
    text_len: int = ...
    off: npt.NDArray[np.int32] = ...
    val: npt.NDArray[np.int32] = ...
    txt: list[bytes] = ...
    n: int = ...
    len: int = ...
    def __init__(
        self, catarray: pd.Series, encoding: Literal["latin-1", "utf-8"] = ...
    ) -> None: ...
    def generate_value_label(self, byteorder: str) -> bytes: ...

class StataMissingValue:
    MISSING_VALUES: dict[float, str] = ...
    bases: tuple[int, int, int] = ...
    float32_base: bytes = ...
    increment: int = ...
    int_value: int = ...
    float64_base: bytes = ...
    BASE_MISSING_VALUES: dict[str, int] = ...
    def __init__(self, value: int | float) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def string(self) -> str: ...
    @property
    def value(self) -> int | float: ...
    @classmethod
    def get_base_missing_value(cls, dtype): ...

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
    def value_labels(self) -> dict[str, dict[int | float, str]]: ...

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
        value_labels: dict[Hashable, dict[float | int, str]] | None = ...,
    ) -> None: ...
    def write_file(self) -> None: ...

class StataStrLWriter:
    df: DataFrame = ...
    columns: Sequence[str] = ...
    def __init__(
        self,
        df: DataFrame,
        columns: Sequence[str],
        version: int = ...,
        byteorder: str | None = ...,
    ) -> None: ...
    def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]: ...
    def generate_blob(self, gso_table: dict[str, tuple[int, int]]) -> bytes: ...

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
        value_labels: dict[Hashable, dict[float | int, str]] | None = ...,
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
        value_labels: dict[Hashable, dict[float | int, str]] | None = ...,
    ) -> None: ...
