from typing import (
    Any,
    Self,
)

from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArraySupportsAnyAll
import pyarrow as pa

class ArrowExtensionArray(OpsMixin, ExtensionArraySupportsAnyAll):
    def __init__(self, values: pa.Array[Any] | pa.ChunkedArray[Any] | Self) -> None: ...
    def __arrow_array__(self, type: Any | None = None) -> pa.ChunkedArray[Any]: ...
