from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

class _HtmlFrameParser:
    io = ...
    match = ...
    attrs = ...
    encoding = ...
    displayed_only = ...
    def __init__(self, io, match, attrs, encoding, displayed_only) -> None: ...
    def parse_tables(self): ...

class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...

class _LxmlFrameParser(_HtmlFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...

def read_html(
    io: FilePathOrBuffer,
    match: str = ...,
    flavor: str | None = ...,
    header: int | Sequence[int] | None = ...,
    index_col: int | Sequence[Any] | None = ...,
    skiprows: int | Sequence[Any] | slice | None = ...,
    attrs: Mapping[str, str] | None = ...,
    parse_dates: bool
    | Sequence[int | str | Sequence[int | str]]
    | dict[str, Sequence[int | str]] = ...,
    thousands: str = ...,
    encoding: str | None = ...,
    decimal: str = ...,
    converters: Mapping[int | str, Callable] | None = ...,
    na_values: Iterable[Any] | None = ...,
    keep_default_na: bool = ...,
    displayed_only: bool = ...,
) -> list[DataFrame]: ...
