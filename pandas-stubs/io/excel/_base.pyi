from types import TracebackType
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    overload,
)

from odf.opendocument import OpenDocument
from openpyxl.workbook.workbook import Workbook
from pandas.core.frame import DataFrame
import pyxlsb.workbook
from xlrd.book import Book

from pandas._typing import (
    Dtype,
    FilePath,
    ReadBuffer,
    StorageOptions,
    WriteExcelBuffer,
)

@overload
def read_excel(
    io: FilePath
    | ReadBuffer[bytes]
    | bytes
    | ExcelFile
    | Workbook
    | Book
    | OpenDocument
    | pyxlsb.workbook.Workbook,
    sheet_name: list[int | str] | None,
    *,
    header: int | Sequence[int] | None = ...,
    names: list[str] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    usecols: Sequence[int] | Sequence[str] | Callable[[str], bool] | None = ...,
    dtype: str | Dtype | dict[str, str | Dtype] | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb"] | None = ...,
    converters: dict[int | str, Callable[[object], object]] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: int | Sequence[int] | Callable[[object], bool] | None = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | dict[str | int, Sequence[str]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: bool
    | Sequence[int]
    | Sequence[Sequence[str] | Sequence[int]]
    | dict[str, Sequence[int] | list[str]] = ...,
    date_parser: Callable | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
) -> dict[int | str, DataFrame]: ...
@overload
def read_excel(
    io: FilePath
    | ReadBuffer[bytes]
    | bytes
    | ExcelFile
    | Workbook
    | Book
    | OpenDocument
    | pyxlsb.workbook.Workbook,
    sheet_name: int | str = ...,
    *,
    header: int | Sequence[int] | None = ...,
    names: list[str] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    usecols: Sequence[int] | Sequence[str] | Callable[[str], bool] | None = ...,
    dtype: str | Dtype | dict[str, str | Dtype] | None = ...,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb"] | None = ...,
    converters: dict[int | str, Callable[[object], object]] | None = ...,
    true_values: Iterable[Hashable] | None = ...,
    false_values: Iterable[Hashable] | None = ...,
    skiprows: int | Sequence[int] | Callable[[object], bool] | None = ...,
    nrows: int | None = ...,
    na_values: Sequence[str] | dict[str | int, Sequence[str]] = ...,
    keep_default_na: bool = ...,
    na_filter: bool = ...,
    verbose: bool = ...,
    parse_dates: bool
    | Sequence[int]
    | Sequence[Sequence[str] | Sequence[int]]
    | dict[str, Sequence[int] | list[str]] = ...,
    date_parser: Callable | None = ...,
    thousands: str | None = ...,
    decimal: str = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame: ...

class ExcelWriter:
    def __init__(
        self,
        path: FilePath | WriteExcelBuffer | ExcelWriter,
        engine: Literal["auto", "openpyxl", "pyxlsb", "odf"] | None = ...,
        date_format: str | None = ...,
        datetime_format: str | None = ...,
        mode: Literal["w", "a"] = ...,
        storage_options: StorageOptions = ...,
        if_sheet_exists: Literal["error", "new", "replace", "overlay"] | None = ...,
        engine_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    @property
    def supported_extensions(self) -> tuple[str, ...]: ...
    @property
    def engine(self) -> Literal["openpyxl", "pyxlsb", "odf"]: ...
    @property
    def sheets(self) -> dict[str, Any]: ...
    @property
    def book(self) -> Workbook | OpenDocument | pyxlsb.workbook.Workbook: ...
    @property
    def date_format(self) -> str: ...
    @property
    def datetime_format(self) -> str: ...
    @property
    def if_sheet_exists(self) -> Literal["error", "new", "replace", "overlay"]: ...
    def __fspath__(self) -> str: ...
    def __enter__(self) -> ExcelWriter: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...

class ExcelFile:
    engine = ...
    io: FilePath | ReadBuffer[bytes] | bytes = ...
    def __init__(
        self,
        io: FilePath | ReadBuffer[bytes] | bytes,
        engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb"] | None = ...,
        storage_options: StorageOptions = ...,
    ) -> None: ...
    def __fspath__(self): ...
    @overload
    def parse(
        self,
        sheet_name: list[int | str] | None,
        header: int | Sequence[int] | None = ...,
        names: list[str] | None = ...,
        index_col: int | Sequence[int] | None = ...,
        usecols: str
        | Sequence[int]
        | Sequence[str]
        | Callable[[str], bool]
        | None = ...,
        converters: dict[int | str, Callable[[object], object]] | None = ...,
        true_values: Iterable[Hashable] | None = ...,
        false_values: Iterable[Hashable] | None = ...,
        skiprows: int | Sequence[int] | Callable[[object], bool] | None = ...,
        nrows: int | None = ...,
        na_values: Sequence[str] | dict[str | int, Sequence[str]] = ...,
        parse_dates: bool
        | Sequence[int]
        | Sequence[Sequence[str] | Sequence[int]]
        | dict[str, Sequence[int] | list[str]] = ...,
        date_parser: Callable | None = ...,
        thousands: str | None = ...,
        comment: str | None = ...,
        skipfooter: int = ...,
        keep_default_na: bool = ...,
        na_filter: bool = ...,
        **kwds: Any,
    ) -> dict[int | str, DataFrame]: ...
    @overload
    def parse(
        self,
        sheet_name: int | str,
        header: int | Sequence[int] | None = ...,
        names: list[str] | None = ...,
        index_col: int | Sequence[int] | None = ...,
        usecols: str
        | Sequence[int]
        | Sequence[str]
        | Callable[[str], bool]
        | None = ...,
        converters: dict[int | str, Callable[[object], object]] | None = ...,
        true_values: Iterable[Hashable] | None = ...,
        false_values: Iterable[Hashable] | None = ...,
        skiprows: int | Sequence[int] | Callable[[object], bool] | None = ...,
        nrows: int | None = ...,
        na_values: Sequence[str] | dict[str | int, Sequence[str]] = ...,
        parse_dates: bool
        | Sequence[int]
        | Sequence[Sequence[str] | Sequence[int]]
        | dict[str, Sequence[int] | list[str]] = ...,
        date_parser: Callable | None = ...,
        thousands: str | None = ...,
        comment: str | None = ...,
        skipfooter: int = ...,
        keep_default_na: bool = ...,
        na_filter: bool = ...,
        **kwds: Any,
    ) -> DataFrame: ...
    @property
    def book(self) -> Workbook | Book | OpenDocument | pyxlsb.workbook.Workbook: ...
    @property
    def sheet_names(self) -> list[int | str]: ...
    def close(self) -> None: ...
    def __enter__(self) -> ExcelFile: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def __del__(self) -> None: ...
