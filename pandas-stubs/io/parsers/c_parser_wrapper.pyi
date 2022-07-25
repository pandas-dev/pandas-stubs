from typing import (
    Hashable,
    Mapping,
    Sequence,
)

from pandas import (
    Index,
    MultiIndex,
)

from pandas._typing import (
    ArrayLike,
    DtypeArg,
    DtypeObj,
    ReadCsvBuffer,
)

from pandas.io.parsers.base_parser import ParserBase

class CParserWrapper(ParserBase):
    low_memory: bool
    kwds = ...  # Incomplete
    unnamed_cols = ...  # Incomplete
    names = ...  # Incomplete
    orig_names = ...  # Incomplete
    index_names = ...  # Incomplete
    def __init__(self, src: ReadCsvBuffer[str], **kwds) -> None: ...
    def close(self) -> None: ...
    def read(
        self, nrows: int | None = ...
    ) -> tuple[
        Index | MultiIndex | None,
        Sequence[Hashable] | MultiIndex,
        Mapping[Hashable, ArrayLike],
    ]: ...

def ensure_dtype_objs(
    dtype: DtypeArg | dict[Hashable, DtypeArg] | None
) -> DtypeObj | dict[Hashable, DtypeObj] | None: ...
