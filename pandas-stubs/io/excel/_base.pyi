from typing import (
    Callable,
    Hashable,
    Iterable,
    Literal,
    Sequence,
    overload,
)

from pandas.core.frame import DataFrame

from pandas._typing import (
    Dtype,
    FilePath,
    IntStrT,
    ReadBuffer,
    StorageOptions,
)

@overload
def read_excel(
    io: FilePath | ReadBuffer[bytes] | bytes,
    sheet_name: list[IntStrT] | None,
    header: int | Sequence[int] | None = ...,
    names: list[str] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    usecols: str | Sequence[int] | Sequence[str] | Callable[[str], bool] | None = ...,
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
    filepath: FilePath | ReadBuffer[bytes] | bytes,
    sheet_name: int | str = ...,
    header: int | Sequence[int] | None = ...,
    names: list[str] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    usecols: str | Sequence[int] | Sequence[str] | Callable[[str], bool] | None = ...,
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
    **kwargs,
) -> DataFrame: ...

class ExcelWriter:
    def __new__(cls, path, engine=..., **kwargs): ...
    book = ...
    curr_sheet = ...
    path = ...
    @property
    def supported_extensions(self): ...
    @property
    def engine(self): ...
    def write_cells(
        self,
        cells,
        sheet_name=...,
        startrow: int = ...,
        startcol: int = ...,
        freeze_panes=...,
    ): ...
    def save(self): ...
    sheets = ...
    cur_sheet = ...
    date_format: str = ...
    datetime_format: str = ...
    mode = ...
    def __init__(
        self,
        path,
        engine=...,
        date_format=...,
        datetime_format=...,
        mode: str = ...,
        **engine_kwargs,
    ) -> None: ...
    def __fspath__(self): ...
    @classmethod
    def check_extension(cls, ext): ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def close(self): ...

class ExcelFile:
    engine = ...
    io = ...
    def __init__(self, io, engine=...) -> None: ...
    def __fspath__(self): ...
    def parse(
        self,
        sheet_name: int = ...,
        header: int = ...,
        names=...,
        index_col=...,
        usecols=...,
        squeeze: bool = ...,
        converters=...,
        true_values=...,
        false_values=...,
        skiprows=...,
        nrows=...,
        na_values=...,
        parse_dates: bool = ...,
        date_parser=...,
        thousands=...,
        comment=...,
        skipfooter: int = ...,
        convert_float: bool = ...,
        mangle_dupe_cols: bool = ...,
        **kwds,
    ): ...
    @property
    def book(self): ...
    @property
    def sheet_names(self): ...
    def close(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def __del__(self) -> None: ...
