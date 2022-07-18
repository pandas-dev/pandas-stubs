from __future__ import annotations

import abc
from typing import (
    Any,
    Callable,
    Sequence,
    overload,
)

from pandas.core.frame import DataFrame

from pandas._typing import (
    Dtype,
    Scalar,
)

@overload
def read_excel(
    filepath: str,
    sheet_name: list[int | str] | None,
    header: int | Sequence[int] | None = ...,
    names: list[str] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    usecols: int | str | Sequence[int | str | Callable] | None = ...,
    squeeze: bool = ...,
    dtype: str | dict[str, Any] | Dtype = ...,
    engine: str | None = ...,
    converters: dict[int | str, Callable] | None = ...,
    true_values: Sequence[Scalar] | None = ...,
    false_values: Sequence[Scalar] | None = ...,
    skiprows: Sequence[int] | int | Callable | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    verbose: bool = ...,
    parse_dates: bool | Sequence | dict[str, Sequence] = ...,
    date_parser: Callable | None = ...,
    thousands: str | None = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    convert_float: bool = ...,
    mangle_dupe_cols: bool = ...,
) -> dict[int | str, DataFrame]: ...
@overload
def read_excel(
    filepath: str,
    sheet_name: int | str = ...,
    header: int | Sequence[int] | None = ...,
    names: list[str] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    usecols: int | str | Sequence[int | str | Callable] | None = ...,
    squeeze: bool = ...,
    dtype: str | dict[str, Any] | Dtype = ...,
    engine: str | None = ...,
    converters: dict[int | str, Callable] | None = ...,
    true_values: Sequence[Scalar] | None = ...,
    false_values: Sequence[Scalar] | None = ...,
    skiprows: Sequence[int] | int | Callable | None = ...,
    nrows: int | None = ...,
    na_values=...,
    keep_default_na: bool = ...,
    verbose: bool = ...,
    parse_dates: bool | Sequence | dict[str, Sequence] = ...,
    date_parser: Callable | None = ...,
    thousands: str | None = ...,
    comment: str | None = ...,
    skipfooter: int = ...,
    convert_float: bool = ...,
    mangle_dupe_cols: bool = ...,
    **kwargs,
) -> DataFrame: ...

class BaseExcelReader(metaclass=abc.ABCMeta):
    book = ...
    def __init__(self, filepath_or_buffer) -> None: ...
    @abc.abstractmethod
    def load_workbook(self, filepath_or_buffer): ...
    def close(self) -> None: ...
    @property
    @abc.abstractmethod
    def sheet_names(self): ...
    @abc.abstractmethod
    def get_sheet_by_name(self, name): ...
    @abc.abstractmethod
    def get_sheet_by_index(self, index): ...
    @abc.abstractmethod
    def get_sheet_data(self, sheet, convert_float): ...
    def parse(
        self,
        sheet_name: int = ...,
        header: int = ...,
        names=...,
        index_col=...,
        usecols=...,
        squeeze: bool = ...,
        dtype=...,
        true_values=...,
        false_values=...,
        skiprows=...,
        nrows=...,
        na_values=...,
        verbose: bool = ...,
        parse_dates: bool = ...,
        date_parser=...,
        thousands=...,
        comment=...,
        skipfooter: int = ...,
        convert_float: bool = ...,
        mangle_dupe_cols: bool = ...,
        **kwds,
    ): ...

class ExcelWriter(metaclass=abc.ABCMeta):
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
