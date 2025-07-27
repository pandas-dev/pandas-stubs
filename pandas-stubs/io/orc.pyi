from typing import Any

from pandas import DataFrame

from pandas._libs.lib import _NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
    ReadBuffer,
)

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = None,
    dtype_backend: DtypeBackend | _NoDefaultDoNotUse = "numpy_nullable",
    # TODO type with the correct pyarrow types
    # filesystem: pyarrow.fs.FileSystem | fsspec.spec.AbstractFileSystem
    filesystem: Any | None = None,
    **kwargs: Any,
) -> DataFrame: ...
