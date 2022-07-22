from collections import abc
from typing import (
    IO,
    Hashable,
    Literal,
    Mapping,
    Sequence,
)

from pandas import (
    Index,
    MultiIndex,
)

from pandas._typing import (
    ArrayLike,
    ReadCsvBuffer,
)

from pandas.io.parsers.base_parser import ParserBase

class PythonParser(ParserBase):
    data = ...  # Incomplete
    buf = ...  # Incomplete
    pos: int
    line_pos: int
    skiprows = ...  # Incomplete
    skipfunc = ...  # Incomplete
    skipfooter = ...  # Incomplete
    delimiter = ...  # Incomplete
    quotechar = ...  # Incomplete
    escapechar = ...  # Incomplete
    doublequote = ...  # Incomplete
    skipinitialspace = ...  # Incomplete
    lineterminator = ...  # Incomplete
    quoting = ...  # Incomplete
    skip_blank_lines = ...  # Incomplete
    names_passed = ...  # Incomplete
    has_index_names: bool
    verbose = ...  # Incomplete
    thousands = ...  # Incomplete
    decimal = ...  # Incomplete
    comment = ...  # Incomplete
    columns = ...  # Incomplete
    orig_names = ...  # Incomplete
    index_names = ...  # Incomplete
    num = ...  # Incomplete
    def __init__(self, f: ReadCsvBuffer[str] | list, **kwds): ...
    def read(
        self, rows: int | None = ...
    ) -> tuple[
        Index | None,
        Sequence[Hashable] | MultiIndex,
        Hashable | ArrayLike,
    ]: ...
    def get_chunk(
        self, size: int | None = ...
    ) -> tuple[
        Index | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, ArrayLike],
    ]: ...

class FixedWidthReader(abc.Iterator):
    f = ...  # Incomplete
    buffer = ...  # Incomplete
    delimiter = ...  # Incomplete
    comment = ...  # Incomplete
    colspecs = ...  # Incomplete
    def __init__(
        self,
        f: IO[str] | ReadCsvBuffer[str],
        colspecs: list[tuple[int, int]] | Literal["infer"],
        delimiter: str | None,
        comment: str | None,
        skiprows: set[int] | None = ...,
        infer_nrows: int = ...,
    ) -> None: ...
    def get_rows(
        self, infer_nrows: int, skiprows: set[int] | None = ...
    ) -> list[str]: ...
    def detect_colspecs(
        self, infer_nrows: int = ..., skiprows: set[int] | None = ...
    ) -> list[tuple[int, int]]: ...
    def __next__(self) -> list[str]: ...

class FixedWidthFieldParser(PythonParser):
    colspecs = ...  # Incomplete
    infer_nrows = ...  # Incomplete
    def __init__(self, f: ReadCsvBuffer[str], **kwds) -> None: ...

def count_empty_vals(vals) -> int: ...
