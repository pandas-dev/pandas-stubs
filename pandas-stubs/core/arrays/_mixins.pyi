from pandas.core.arrays.base import ExtensionArray
from pandas.core.series import Series
from typing_extensions import Self

from pandas._libs.arrays import NDArrayBacked
from pandas._typing import (
    AxisInt,
    Scalar,
)

class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):
    @property
    def shape(self) -> tuple[int]: ...
    def argmin(self, axis: AxisInt = 0, skipna: bool = True) -> int: ...
    def argmax(self, axis: AxisInt = 0, skipna: bool = True) -> int: ...
    def insert(self, loc: int, item: Scalar) -> Self: ...
    def value_counts(self, dropna: bool = True) -> Series[int]: ...
