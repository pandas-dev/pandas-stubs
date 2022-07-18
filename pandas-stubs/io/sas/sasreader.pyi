from __future__ import annotations

from typing import (
    Optional,
    Sequence,
)

from pandas._typing import FilePathOrBuffer

def read_sas(
    path: FilePathOrBuffer,
    format: str | None = ...,
    index: Sequence | None = ...,
    encoding: str | None = ...,
    chunksize: int | None = ...,
    iterator: bool = ...,
): ...
