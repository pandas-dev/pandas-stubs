from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
import csv
from typing import (
    Any,
    Literal,
    overload,
)

from pandas.core.frame import DataFrame

from pandas._libs.lib import NoDefault
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    CSVQuoting,
    DtypeArg,
    DtypeBackend,
    HashableT,
    ListLikeHashable,
    StorageOptions,
    UsecolsArgType,
)

from pandas.io.parsers import TextFileReader

@overload
def read_clipboard(
    sep: str | None = r"\s+",
    *,
    dtype_backend: DtypeBackend | NoDefault = ...,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    dtype: DtypeArg | Mapping[HashableT, DtypeArg] | None = None,
    engine: CSVEngine | None = None,
    converters: dict[int | str, Callable[[str], Any]] | None = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | Sequence[int]
        | list[str]
        | Sequence[Sequence[int]]
        | dict[str, Sequence[int]]
        | None
    ) = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: Literal[True],
    chunksize: int | None = None,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
) -> TextFileReader: ...
@overload
def read_clipboard(
    sep: str | None = r"\s+",
    *,
    dtype_backend: DtypeBackend | NoDefault = ...,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    dtype: DtypeArg | Mapping[HashableT, DtypeArg] | None = None,
    engine: CSVEngine | None = None,
    converters: dict[int | str, Callable[[str], Any]] | None = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | Sequence[int]
        | list[str]
        | Sequence[Sequence[int]]
        | dict[str, Sequence[int]]
        | None
    ) = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: bool = False,
    chunksize: int,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
) -> TextFileReader: ...
@overload
def read_clipboard(
    sep: str | None = r"\s+",
    *,
    dtype_backend: DtypeBackend | NoDefault = ...,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: ListLikeHashable | None = ...,
    index_col: int | str | Sequence[str | int] | Literal[False] | None = None,
    usecols: UsecolsArgType = None,
    dtype: DtypeArg | Mapping[HashableT, DtypeArg] | None = None,
    engine: CSVEngine | None = None,
    converters: dict[int | str, Callable[[str], Any]] | None = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable[[int], bool] | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | dict[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: (
        bool
        | Sequence[int]
        | list[str]
        | Sequence[Sequence[int]]
        | dict[str, Sequence[int]]
        | None
    ) = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: Literal[False] = False,
    chunksize: None = None,
    compression: CompressionOptions = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: CSVQuoting = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | csv.Dialect | None = None,
    on_bad_lines: (
        Literal["error", "warn", "skip"] | Callable[[list[str]], list[str] | None]
    ) = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: StorageOptions | None = None,
) -> DataFrame: ...
