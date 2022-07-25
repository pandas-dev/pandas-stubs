from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pandas import DataFrame

from pandas._typing import (
    FilePath,
    ReadBuffer,
    WriteBuffer,
)

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[str] | None = ...,
    **kwargs: Any,
) -> DataFrame: ...
def to_orc(
    df: DataFrame,
    path: FilePath | WriteBuffer[bytes] | None = ...,
    *,
    engine: Literal["pyarrow"] = ...,
    index: bool | None = ...,
    engine_kwargs: dict[str, Any] | None = ...,
) -> bytes | None: ...
