from __future__ import annotations

from typing import (
    Any,
    Pattern,
    Sequence,
)

from pandas.core.frame import DataFrame

from pandas._typing import (
    FilePath,
    ReadBuffer,
)

class _HtmlFrameParser:
    io: FilePath | ReadBuffer[str] | ReadBuffer[bytes]
    match: str | Pattern
    attrs: dict[str, str] | None
    encoding: str
    displayed_only: bool
    def __init__(
        self,
        io: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
        match: str | Pattern,
        attrs: dict[str, str] | None,
        encoding: str,
        displayed_only: bool,
    ) -> None: ...
    def parse_tables(self): ...

class _BeautifulSoupHtml5LibFrameParser(_HtmlFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...

class _LxmlFrameParser(_HtmlFrameParser):
    def __init__(self, *args, **kwargs) -> None: ...

def read_html(
    io: FilePath | ReadBuffer[str],
    match: str | Pattern = ...,
    flavor: str | None = ...,
    header: int | Sequence[int] | None = ...,
    index_col: int | Sequence[int] | None = ...,
    skiprows: int | Sequence[int] | slice | None = ...,
    attrs: dict[str, str] | None = ...,
    parse_dates: bool = ...,
    thousands: str | None = ...,
    encoding: str | None = ...,
    decimal: str = ...,
    converters: dict | None = ...,
    na_values: list[Any] | None = ...,
    keep_default_na: bool = ...,
    displayed_only: bool = ...,
) -> list[DataFrame]: ...
