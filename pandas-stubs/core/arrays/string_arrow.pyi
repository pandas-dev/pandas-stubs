from typing import Literal

from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.string_ import BaseStringArray

class ArrowStringArray(ArrowExtensionArray, BaseStringArray[Literal["pyarrow"]]): ...
