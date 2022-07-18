from __future__ import annotations

from typing import (
    List,
    Optional,
)

from pandas.core.frame import DataFrame

from pandas._typing import FilePathOrBuffer

def read_orc(
    path: FilePathOrBuffer, columns: list[str] | None = ..., **kwargs
) -> DataFrame: ...
