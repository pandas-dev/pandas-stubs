from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pandas.core.frame import DataFrame

from pandas._typing import (
    FilePath,
    ReadBuffer,
    WriteBuffer,
)

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[str] | None = ...,
    **kwargs,
) -> DataFrame: ...
def to_orc(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] = ...,
    *,
    engine: Literal["pyarrow"] = ...,
    index: bool | None = ...,
    engine_kwargs: dict[str, Any] | None = ...,
) -> bytes | None: ...
