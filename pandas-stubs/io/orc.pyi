from typing import Any

from fsspec import (  # pyright: ignore[reportMissingTypeStubs]
    AbstractFileSystem,
)
from pandas import DataFrame
from pyarrow.fs import FileSystem

from pandas._libs.lib import NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
    ReadBuffer,
)

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = None,
    dtype_backend: DtypeBackend | NoDefaultDoNotUse = "numpy_nullable",
    filesystem: FileSystem | AbstractFileSystem | None = None,
    **kwargs: Any,
) -> DataFrame: ...
