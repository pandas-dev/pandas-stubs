from typing import Any

from pandas.core.arraylike import OpsMixin
from pandas.core.arrays.base import ExtensionArraySupportsAnyAll
import pyarrow as pa
from typing_extensions import Self

class ArrowExtensionArray(OpsMixin, ExtensionArraySupportsAnyAll):
    def __new__(cls, values: pa.Array[Any] | pa.ChunkedArray[Any]) -> Self: ...
