import abc
from pandas._typing import Dtype, Scalar
from pandas.core.frame import DataFrame as DataFrame
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    overload,
)

@overload
def read_excel(
    filepath: str,
    sheet_name: List[Union[int, str]],
    header: Optional[Union[int, Sequence[int]]] = ...,
    names: Optional[List[str]] = ...,
    index_col: Optional[Union[int, Sequence[int]]] = ...,
    usecols: Optional[Union[int, str, Sequence[Union[int, str, Callable]]]] = ...,
    squeeze: bool = ...,
    dtype: Union[str, Dict[str, Any], Dtype] = ...,
    engine: Optional[str] = ...,
    converters: Optional[Dict[Union[int, str], Callable]] = ...,
    true_values: Optional[Sequence[Scalar]] = ...,
    false_values: Optional[Sequence[Scalar]] = ...,
    skiprows: Optional[Union[Sequence[int], int, Callable]] = ...,
    nrows: Optional[int] = ...,
    na_values=...,
    keep_default_na: bool = ...,
    verbose: bool = ...,
    parse_dates: Union[bool, Sequence, Dict[str, Sequence]] = ...,
    date_parser: Optional[Callable] = ...,
    thousands: Optional[str] = ...,
    comment: Optional[str] = ...,
    skipfooter: int = ...,
    convert_float: bool = ...,
    mangle_dupe_cols: bool = ...,
) -> Dict[Union[int, str], DataFrame]: ...
@overload
def read_excel(
    filepath: str,
    sheet_name: Union[int, str] = ...,
    header: Optional[Union[int, Sequence[int]]] = ...,
    names: Optional[List[str]] = ...,
    index_col: Optional[Union[int, Sequence[int]]] = ...,
    usecols: Optional[Union[int, str, Sequence[Union[int, str, Callable]]]] = ...,
    squeeze: bool = ...,
    dtype: Union[str, Dict[str, Any], Dtype] = ...,
    engine: Optional[str] = ...,
    converters: Optional[Dict[Union[int, str], Callable]] = ...,
    true_values: Optional[Sequence[Scalar]] = ...,
    false_values: Optional[Sequence[Scalar]] = ...,
    skiprows: Optional[Union[Sequence[int], int, Callable]] = ...,
    nrows: Optional[int] = ...,
    na_values=...,
    keep_default_na: bool = ...,
    verbose: bool = ...,
    parse_dates: Union[bool, Sequence, Dict[str, Sequence]] = ...,
    date_parser: Optional[Callable] = ...,
    thousands: Optional[str] = ...,
    comment: Optional[str] = ...,
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
