import abc
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Iterable,
    Mapping,
    Sequence,
)

from pandas.core.frame import DataFrame
from pandas.core.indexes.api import Index
from pandas.core.series import Series

from pandas._config import get_option as get_option

from pandas._typing import (
    Dtype,
    WriteBuffer,
)

from pandas.io.formats.printing import pprint_thing as pprint_thing

frame_max_cols_sub = ...  # Incomplete
show_counts_sub = ...  # Incomplete
null_counts_sub = ...  # Incomplete
frame_examples_sub = ...  # Incomplete
frame_see_also_sub = ...  # Incomplete
frame_sub_kwargs = ...  # Incomplete
series_examples_sub = ...  # Incomplete
series_see_also_sub = ...  # Incomplete
series_sub_kwargs = ...  # Incomplete
INFO_DOCSTRING = ...  # Incomplete

class BaseInfo(ABC, metaclass=abc.ABCMeta):
    data: DataFrame | Series
    memory_usage: bool | str
    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]: ...
    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]: ...
    @property
    @abstractmethod
    def memory_usage_bytes(self) -> int: ...
    @property
    def memory_usage_string(self) -> str: ...
    @property
    def size_qualifier(self) -> str: ...
    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None: ...

class DataFrameInfo(BaseInfo):
    data: DataFrame | Series
    memory_usage: bool | str
    def __init__(
        self, data: DataFrame, memory_usage: bool | str | None = ...
    ) -> None: ...
    @property
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    def dtypes(self) -> Iterable[Dtype]: ...
    @property
    def ids(self) -> Index: ...
    @property
    def col_count(self) -> int: ...
    @property
    def non_null_counts(self) -> Sequence[int]: ...
    @property
    def memory_usage_bytes(self) -> int: ...
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None: ...

class SeriesInfo(BaseInfo):
    data: DataFrame | Series
    memory_usage: bool | str
    def __init__(self, data: Series, memory_usage: bool | str | None = ...) -> None: ...
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None = ...,
        max_cols: int | None = ...,
        verbose: bool | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...
    @property
    def non_null_counts(self) -> Sequence[int]: ...
    @property
    def dtypes(self) -> Iterable[Dtype]: ...
    @property
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    def memory_usage_bytes(self) -> int: ...

class InfoPrinterAbstract(metaclass=abc.ABCMeta):
    def to_buffer(self, buf: WriteBuffer[str] | None = ...) -> None: ...

class DataFrameInfoPrinter(InfoPrinterAbstract):
    info = ...  # Incomplete
    data = ...  # Incomplete
    verbose = ...  # Incomplete
    max_cols = ...  # Incomplete
    show_counts = ...  # Incomplete
    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = ...,
        verbose: bool | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...
    @property
    def max_rows(self) -> int: ...
    @property
    def exceeds_info_cols(self) -> bool: ...
    @property
    def exceeds_info_rows(self) -> bool: ...
    @property
    def col_count(self) -> int: ...

class SeriesInfoPrinter(InfoPrinterAbstract):
    info = ...  # Incomplete
    data = ...  # Incomplete
    verbose = ...  # Incomplete
    show_counts = ...  # Incomplete
    def __init__(
        self,
        info: SeriesInfo,
        verbose: bool | None = ...,
        show_counts: bool | None = ...,
    ) -> None: ...

class TableBuilderAbstract(ABC, metaclass=abc.ABCMeta):
    info: BaseInfo
    @abstractmethod
    def get_lines(self) -> list[str]: ...
    @property
    def data(self) -> DataFrame | Series: ...
    @property
    def dtypes(self) -> Iterable[Dtype]: ...
    @property
    def dtype_counts(self) -> Mapping[str, int]: ...
    @property
    def display_memory_usage(self) -> bool: ...
    @property
    def memory_usage_string(self) -> str: ...
    @property
    def non_null_counts(self) -> Sequence[int]: ...
    def add_object_type_line(self) -> None: ...
    def add_index_range_line(self) -> None: ...
    def add_dtypes_line(self) -> None: ...

class DataFrameTableBuilder(TableBuilderAbstract, metaclass=abc.ABCMeta):
    info: BaseInfo
    def __init__(self, *, info: DataFrameInfo) -> None: ...
    def get_lines(self) -> list[str]: ...
    @property
    def data(self) -> DataFrame: ...
    @property
    def ids(self) -> Index: ...
    @property
    def col_count(self) -> int: ...
    def add_memory_usage_line(self) -> None: ...

class DataFrameTableBuilderNonVerbose(DataFrameTableBuilder):
    def add_columns_summary_line(self) -> None: ...

class TableBuilderVerboseMixin(TableBuilderAbstract, metaclass=abc.ABCMeta):
    SPACING: str
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool
    @property
    @abstractmethod
    def headers(self) -> Sequence[str]: ...
    @property
    def header_column_widths(self) -> Sequence[int]: ...
    def add_header_line(self) -> None: ...
    def add_separator_line(self) -> None: ...
    def add_body_lines(self) -> None: ...

class DataFrameTableBuilderVerbose(DataFrameTableBuilder, TableBuilderVerboseMixin):
    info: DataFrameInfo
    with_counts: bool
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    def __init__(self, *, info: DataFrameInfo, with_counts: bool) -> None: ...
    @property
    def headers(self) -> Sequence[str]: ...
    def add_columns_summary_line(self) -> None: ...

class SeriesTableBuilder(TableBuilderAbstract, metaclass=abc.ABCMeta):
    info: BaseInfo
    def __init__(self, *, info: SeriesInfo) -> None: ...
    def get_lines(self) -> list[str]: ...
    @property
    def data(self) -> Series: ...
    def add_memory_usage_line(self) -> None: ...

class SeriesTableBuilderNonVerbose(SeriesTableBuilder): ...

class SeriesTableBuilderVerbose(SeriesTableBuilder, TableBuilderVerboseMixin):
    info: SeriesInfo
    with_counts: bool
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    def __init__(self, *, info: SeriesInfo, with_counts: bool) -> None: ...
    def add_series_name_line(self) -> None: ...
    @property
    def headers(self) -> Sequence[str]: ...
