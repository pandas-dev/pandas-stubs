from typing import Self

from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

from pandas._typing import np_ndarray

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray):
    def __new__(cls, values: np_ndarray | Self, copy: bool = False) -> Self: ...
