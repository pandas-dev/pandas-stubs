from typing import Literal

from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.string_ import BaseStringArray
import pyarrow as pa
from typing_extensions import Self

from pandas._typing import DtypeArg

class ArrowStringArray(ArrowExtensionArray, BaseStringArray[Literal["pyarrow"]]):
    def __new__(cls, values: pa.StringArray, copy: bool = False) -> Self: ...
    def __arrow_array__(self, type: DtypeArg | None = None) -> pa.StringArray: ...
