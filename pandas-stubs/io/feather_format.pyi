from pandas import DataFrame

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
    ReadBuffer,
    StorageOptions,
)

def read_feather(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = ...,
    use_threads: bool = ...,
    storage_options: StorageOptions = ...,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = ...,
) -> DataFrame: ...
