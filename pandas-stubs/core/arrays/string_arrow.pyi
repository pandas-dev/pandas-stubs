from typing import Literal

from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.string_ import BaseStringArray
import pyarrow as pa

class ArrowStringArray(ArrowExtensionArray, BaseStringArray[Literal["pyarrow"]]):
    def __init__(
        self,
        values: pa.StringArray | pa.ChunkedArray[pa.StringScalar],
        copy: bool = False,
    ) -> None: ...
