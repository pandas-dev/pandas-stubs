from collections import abc
import csv
from typing import (
    Any,
    Hashable,
    Literal,
    Sequence,
    overload,
)

from pandas.core.frame import DataFrame

import pandas._libs.lib as lib
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    DtypeArg,
    FilePath,
    IndexLabel,
    ReadCsvBuffer,
    StorageOptions,
)

@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
) -> TextFileReader: ...
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
) -> TextFileReader: ...
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame | TextFileReader: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[True],
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace=...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
) -> TextFileReader: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace=...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
) -> TextFileReader: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: Literal[False] = ...,
    chunksize: None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace=...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None | lib.NoDefault = ...,
    delimiter: str | None | lib.NoDefault = ...,
    header: int | Sequence[int] | None | Literal["infer"] = ...,
    names: Sequence[Hashable] | None | lib.NoDefault = ...,
    index_col: IndexLabel | Literal[False] | None = ...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | lib.NoDefault = ...,
    mangle_dupe_cols: bool = ...,
    dtype: DtypeArg | None = ...,
    engine: CSVEngine | None = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    skip_blank_lines: bool = ...,
    parse_dates=...,
    infer_datetime_format: bool = ...,
    keep_date_col: bool = ...,
    date_parser=...,
    dayfirst: bool = ...,
    cache_dates: bool = ...,
    iterator: bool = ...,
    chunksize: int | None = ...,
    compression: CompressionOptions = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    lineterminator: str | None = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: str | None = ...,
    comment: str | None = ...,
    encoding: str | None = ...,
    encoding_errors: str | None = ...,
    dialect: str | csv.Dialect | None = ...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace=...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: str | None = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame | TextFileReader: ...
def read_fwf(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    colspecs: Sequence[tuple[int, int]] | str | None = ...,
    widths: Sequence[int] | None = ...,
    infer_nrows: int = ...,
    **kwds,
) -> DataFrame | TextFileReader: ...

class TextFileReader(abc.Iterator):
    engine: Any  # Incomplete
    orig_options: Any  # Incomplete
    chunksize: Any  # Incomplete
    nrows: Any  # Incomplete
    squeeze: Any  # Incomplete
    handles: Any  # Incomplete
    def __init__(
        self,
        f: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str] | list,
        engine: CSVEngine | None = ...,
        **kwds: Any,
    ) -> None: ...
    def close(self) -> None: ...
    def read(self, nrows: int | None = ...) -> DataFrame: ...
    def get_chunk(self, size: int | None = ...) -> DataFrame: ...
    def __next__(self) -> DataFrame: ...
    def __enter__(self) -> TextFileReader: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
