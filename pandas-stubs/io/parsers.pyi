from __future__ import annotations

from collections import abc
from typing import (
    Callable,
    Literal,
    Protocol,
    Sequence,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from pandas._typing import (
    AnyStr_cov,
    CompressionOptions,
    DtypeArg,
    FilePath,
    FilePathOrBuffer,
    ReadBuffer,
    StorageOptions,
)

ListLike = Union[
    list[Union[str, int]],
    tuple[Union[str, int]],
    set[Union[str, int]],
    np.ndarray,
    pd.Series,
]

class ReadCsvBuffer(ReadBuffer[AnyStr_cov], Protocol): ...

# read_csv engines
CSVEngine = Literal["c", "python", "pyarrow", "python-fwf"]

# iterator=True -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols: ListLike | Callable | None = ...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> TextFileReader: ...

# chunksize=int -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols: ListLike | Callable | None = ...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> TextFileReader: ...

# default case -> DataFrame
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols: ListLike | Callable | None = ...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> DataFrame: ...

# Unions -> DataFrame | TextFileReader
@overload
def read_csv(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols: ListLike | Callable | None = ...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> DataFrame | TextFileReader: ...

# iterator=True -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> TextFileReader: ...

# chunksize=int -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> TextFileReader: ...

# default case -> DataFrame
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> DataFrame: ...

# Unions -> DataFrame | TextFileReader
@overload
def read_table(
    filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    sep: str | None = ...,
    delimiter: str | None = ...,
    header: int | Sequence[int] | Literal["infer"] | None = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: bool | None = ...,
    prefix: str | None = ...,
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
    dialect=...,
    error_bad_lines: bool | None = ...,
    warn_bad_lines: bool | None = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Literal["high", "legacy"] | None = ...,
    storage_options: StorageOptions | None = ...,
) -> DataFrame | TextFileReader: ...
def read_fwf(
    filepath_or_buffer: FilePathOrBuffer,
    colspecs=...,
    widths=...,
    infer_nrows=...,
    **kwds,
): ...

class TextFileReader(abc.Iterator):
    f = ...
    orig_options = ...
    engine = ...
    chunksize = ...
    nrows = ...
    squeeze = ...
    def __init__(self, f, engine=..., **kwds) -> None: ...
    def close(self) -> None: ...
    def __next__(self): ...
    def read(self, nrows=...): ...
    def get_chunk(self, size=...): ...

class ParserBase:
    names = ...
    orig_names = ...
    prefix = ...
    index_col = ...
    unnamed_cols = ...
    index_names = ...
    col_names = ...
    parse_dates = ...
    date_parser = ...
    dayfirst = ...
    keep_date_col = ...
    na_values = ...
    na_fvalues = ...
    na_filter = ...
    keep_default_na = ...
    true_values = ...
    false_values = ...
    mangle_dupe_cols = ...
    infer_datetime_format = ...
    cache_dates = ...
    header = ...
    handles = ...
    def __init__(self, kwds) -> None: ...
    def close(self) -> None: ...

class CParserWrapper(ParserBase):
    kwds = ...
    unnamed_cols = ...
    names = ...
    orig_names = ...
    index_names = ...
    def __init__(self, src, **kwds) -> None: ...
    def close(self) -> None: ...
    def set_error_bad_lines(self, status) -> None: ...
    def read(self, nrows=...): ...

def TextParser(*args, **kwds): ...
def count_empty_vals(vals): ...

class PythonParser(ParserBase):
    data = ...
    buf = ...
    pos: int = ...
    line_pos: int = ...
    encoding = ...
    compression = ...
    memory_map = ...
    skiprows = ...
    skipfunc = ...
    skipfooter = ...
    delimiter = ...
    quotechar = ...
    escapechar = ...
    doublequote = ...
    skipinitialspace = ...
    lineterminator = ...
    quoting = ...
    skip_blank_lines = ...
    warn_bad_lines = ...
    error_bad_lines = ...
    names_passed = ...
    has_index_names: bool = ...
    verbose = ...
    converters = ...
    dtype = ...
    thousands = ...
    decimal = ...
    comment = ...
    num_original_columns = ...
    columns = ...
    orig_names = ...
    index_names = ...
    nonnum = ...
    def __init__(self, f, **kwds): ...
    def read(self, rows=...): ...
    def get_chunk(self, size=...): ...

class FixedWidthReader(abc.Iterator):
    f = ...
    buffer = ...
    delimiter = ...
    comment = ...
    colspecs = ...
    def __init__(
        self, f, colspecs, delimiter, comment, skiprows=..., infer_nrows: int = ...
    ) -> None: ...
    def get_rows(self, infer_nrows, skiprows=...): ...
    def detect_colspecs(self, infer_nrows: int = ..., skiprows=...): ...
    def __next__(self): ...

class FixedWidthFieldParser(PythonParser):
    colspecs = ...
    infer_nrows = ...
    def __init__(self, f, **kwds) -> None: ...
