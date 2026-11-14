from typing import (
    Generic,
    Literal,
)

from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.string_ import (
    BaseStringArray,
)
import pyarrow as pa
from typing_extensions import TypeVar

from pandas._libs.missing import NAType

_ArrowNaValueT = TypeVar(
    "_ArrowNaValueT", bound=NAType | float, covariant=True, default=NAType
)

class ArrowStringArray(
    ArrowExtensionArray,
    BaseStringArray[Literal["pyarrow"], _ArrowNaValueT],
    Generic[_ArrowNaValueT],
):
    def __init__(
        self,
        values: pa.StringArray | pa.ChunkedArray[pa.StringScalar],
        copy: bool = False,
    ) -> None: ...
