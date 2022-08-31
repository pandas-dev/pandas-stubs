from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Pattern,
    Sequence,
)

from pandas.core.frame import DataFrame

from pandas._typing import (
    FilePath,
    HashableT,
    ReadBuffer,
)

def read_html(
    io: FilePath | ReadBuffer[str],
    match: str | Pattern = ...,
    flavor: str | None = ...,
    header: int | Sequence[int] | None = ...,
    index_col: int | Sequence[int] | list[HashableT] | None = ...,
    skiprows: int | Sequence[int] | slice | None = ...,
    attrs: dict[str, str] | None = ...,
    parse_dates: bool
    | Sequence[int]
    | list[HashableT]  # Cannot be Sequence[Hashable] to prevent str
    | Sequence[Sequence[Hashable]]
    | dict[str, Sequence[int]]
    | dict[str, list[HashableT]] = ...,
    thousands: str = ...,
    encoding: str | None = ...,
    decimal: str = ...,
    converters: Mapping[int | HashableT, Callable[[str], Any]] | None = ...,
    na_values: str
    | list[str]
    | dict[HashableT, str]
    | dict[HashableT, list[str]]
    | None = ...,
    keep_default_na: bool = ...,
    displayed_only: bool = ...,
    extract_links: Literal["header", "footer", "body", "all"] | None = ...,
) -> list[DataFrame]: ...
