from typing import (
    Any,
    Self,
)

from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArraySupportsAnyAll
import pyarrow as pa

from pandas._typing import (
    NpDtype,
    np_1darray,
)

class ArrowExtensionArray(OpsMixin, ExtensionArraySupportsAnyAll):
    def __init__(self, values: pa.Array[Any] | pa.ChunkedArray[Any] | Self) -> None: ...
    def __arrow_array__(self, type: Any | None = None) -> pa.ChunkedArray[Any]: ...
    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np_1darray: ...
