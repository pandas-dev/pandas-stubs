from collections.abc import Sequence
from typing import (
    Any,
    TypeAlias,
)

from pandas import DataFrame

from pandas._typing import (
    DtypeBackend,
    FilePath,
    ParquetEngine,
    ReadBuffer,
    StorageOptions,
)

_Filter: TypeAlias = tuple[str, str, Any]

def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: ParquetEngine = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend = "numpy_nullable",
    filesystem: Any = None,
    filters: Sequence[_Filter] | Sequence[Sequence[_Filter]] | None = None,
    **kwargs: Any,
) -> DataFrame: ...
