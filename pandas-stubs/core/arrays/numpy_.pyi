from pandas.core.arraylike import OpsMixin
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from typing_extensions import Self

from pandas._typing import np_ndarray

class NumpyExtensionArray(OpsMixin, NDArrayBackedExtensionArray):
    def __new__(cls, values: np_ndarray | Self, copy: bool = False) -> Self: ...
