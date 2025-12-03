from typing import (
    Any,
    Literal,
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

_OP: TypeAlias = Literal["==", "=", ">", ">=", "<", "<=", "!=", "in", "not in"]
_FILTER: TypeAlias = tuple[str, _OP, Any]

def read_parquet(
    path: FilePath | ReadBuffer[bytes],
    engine: ParquetEngine = "auto",
    columns: list[str] | None = None,
    storage_options: StorageOptions = None,
    dtype_backend: DtypeBackend = "numpy_nullable",
    filesystem: Any = None,
    filters: list[_FILTER] | list[list[_FILTER]] | None = None,
    **kwargs: Any,
) -> DataFrame: ...
