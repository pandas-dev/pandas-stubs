from collections.abc import Sequence
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

from pandas import DataFrame

from pandas._typing import (
    DtypeBackend,
    FilePath,
    ReadBuffer,
    StorageOptions,
)

_Filter: TypeAlias = tuple[str, str, Any]

@overload
def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: Literal["auto", "fastparquet"] = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend = "numpy_nullable",
    filesystem: Any = None,
    filters: Sequence[_Filter] | Sequence[Sequence[_Filter]] | None = None,
    to_pandas_kwargs: None = None,
    **kwargs: Any,
) -> DataFrame: ...
@overload
def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: Literal["pyarrow"],
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend = "numpy_nullable",
    filesystem: Any = None,
    filters: Sequence[_Filter] | Sequence[Sequence[_Filter]] | None = None,
    *,
    to_pandas_kwargs: dict[str, Any],
    **kwargs: Any,
) -> DataFrame: ...
