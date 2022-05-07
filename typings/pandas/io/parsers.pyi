from __future__ import annotations
from collections import abc
import sys
from pandas._typing import (
    FilePath as FilePath,
    FilePathOrBuffer as FilePathOrBuffer,
    Scalar as Scalar,
    ReadBuffer as ReadBuffer,
    AnyStr_cov as AnyStr_cov,
    DtypeArg as DtypeArg,
    CompressionOptions as CompressionOptions,
    StorageOptions as StorageOptions,
)
from pandas.core.frame import DataFrame as DataFrame
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    overload,
    Protocol,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

class ReadCsvBuffer(ReadBuffer[AnyStr_cov], Protocol): ...

# read_csv engines
CSVEngine = Literal["c", "python", "pyarrow", "python-fwf"]

# iterator=True -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    chunksize: Optional[int] = ...,
    compression: CompressionOptions = ...,
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> TextFileReader: ...

# chunksize=int -> TextFileReader
@overload
def read_csv(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> TextFileReader: ...

# default case -> DataFrame
@overload
def read_csv(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> DataFrame: ...

# Unions -> DataFrame | TextFileReader
@overload
def read_csv(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    chunksize: Optional[int] = ...,
    compression: CompressionOptions = ...,
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> DataFrame | TextFileReader: ...

# iterator=True -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    chunksize: Optional[int] = ...,
    compression: CompressionOptions = ...,
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> TextFileReader: ...

# chunksize=int -> TextFileReader
@overload
def read_table(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> TextFileReader: ...

# default case -> DataFrame
@overload
def read_table(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
) -> DataFrame: ...

# Unions -> DataFrame | TextFileReader
@overload
def read_table(
    filepath_or_buffer: Union[FilePath, ReadCsvBuffer[bytes], ReadCsvBuffer[str]],
    *,
    sep: Optional[str] = ...,
    delimiter: Optional[str] = ...,
    header: Optional[Union[int, Sequence[int], Literal["infer"]]] = ...,
    names=...,
    index_col=...,
    usecols=...,
    squeeze: Optional[bool] = ...,
    prefix: Optional[str] = ...,
    mangle_dupe_cols: bool = ...,
    dtype: Optional[DtypeArg] = ...,
    engine: Optional[CSVEngine] = ...,
    converters=...,
    true_values=...,
    false_values=...,
    skipinitialspace: bool = ...,
    skiprows=...,
    skipfooter: int = ...,
    nrows: Optional[int] = ...,
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
    chunksize: Optional[int] = ...,
    compression: CompressionOptions = ...,
    thousands: Optional[str] = ...,
    decimal: str = ...,
    lineterminator: Optional[str] = ...,
    quotechar: str = ...,
    quoting: int = ...,
    doublequote: bool = ...,
    escapechar: Optional[str] = ...,
    comment: Optional[str] = ...,
    encoding: Optional[str] = ...,
    encoding_errors: Optional[str] = ...,
    dialect=...,
    error_bad_lines: Optional[bool] = ...,
    warn_bad_lines: Optional[bool] = ...,
    on_bad_lines=...,
    delim_whitespace: bool = ...,
    low_memory=...,
    memory_map: bool = ...,
    float_precision: Optional[Literal["high", "legacy"]] = ...,
    storage_options: Optional[StorageOptions] = ...,
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
