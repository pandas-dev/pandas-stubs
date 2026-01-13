from typing import Any

from pandas import DataFrame
from pyarrow.fs import FileSystem

from pandas._libs.lib import NoDefaultDoNotUse
from pandas._typing import (
    DtypeBackend,
    FilePath,
    HashableT,
    ReadBuffer,
)

from fsspec.spec import (  # pyright: ignore[reportMissingTypeStubs] # isort: skip
    AbstractFileSystem,  # pyright: ignore[reportUnknownVariableType]
)

def read_orc(
    path: FilePath | ReadBuffer[bytes],
    columns: list[HashableT] | None = None,
    dtype_backend: DtypeBackend | NoDefaultDoNotUse = "numpy_nullable",
    filesystem: (  # pyright: ignore[reportUnknownParameterType]
        FileSystem | AbstractFileSystem | None
    ) = None,
    **kwargs: Any,
) -> DataFrame: ...
